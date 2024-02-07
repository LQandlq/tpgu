# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from pathlib import PurePath
from typing import Any

import gin
import tensorflow as tf
from absl import app
from absl import flags
from absl import logging

import network_trainer
from official.common import distribute_utils
from official.common import flags as tfm_flags
from official.core import task_factory
from official.core import train_lib
from official.core import train_utils
from official.core.train_utils import BestCheckpointExporter
from official.modeling import performance
from official.vision.utils import summary_manager
from gradient_accumulator import GradientAccumulator
import registry_imports  # pylint: disable=unused-import

FLAGS = flags.FLAGS

def may_create_best_ckpt_exporter(data_dir: str, export_subdir: str, metric_name: str, metric_comp: str) -> Any:
    """Maybe create a BestCheckpointExporter object, according to the config."""
    if data_dir and export_subdir and metric_name:
        best_ckpt_dir = str(PurePath(data_dir, export_subdir))
        best_ckpt_exporter = BestCheckpointExporter(best_ckpt_dir, metric_name, metric_comp)
        logging.info('Created the best checkpoint exporter. '
                     f'data_dir: {data_dir}, export_subdir: {export_subdir}, metric_name: {metric_name}')
    else:
        best_ckpt_exporter = None

    return best_ckpt_exporter


def summary_model(params, model_dir):
    task = task_factory.get_task(params.task, logging_dir=model_dir)
    model = task.build_model()
    flops = train_utils.try_count_flops(model)
    if flops is not None:
        logging.info(f'FLOPs (multi-adds) in model: {flops / 10. ** 9 / 2} Billions.')

    model.summary(print_fn=logging.info, expand_nested=True, show_trainable=True)

    del task
    del model


def _run_experiment_with_preemption_recovery(params, model_dir):
    """Runs experiment and tries to reconnect when encounting a preemption."""
    keep_training = True
    while keep_training:
        preemption_watcher = None
        try:
            distribution_strategy = distribute_utils.get_distribution_strategy(
                distribution_strategy=params.runtime.distribution_strategy,
                all_reduce_alg=params.runtime.all_reduce_alg, num_gpus=params.runtime.num_gpus,
                tpu_address=params.runtime.tpu)

            with distribution_strategy.scope():
                task = task_factory.get_task(params.task, logging_dir=model_dir)
                model = task.build_model()
                optimizer = train_utils.create_optimizer(task, params)

                best_ckpt_exporter = may_create_best_ckpt_exporter(
                    model_dir,
                    params.trainer.best_checkpoint_export_subdir,
                    params.trainer.best_checkpoint_eval_metric,
                    params.trainer.best_checkpoint_metric_comp)

                best_ema_ckpt_exporter = may_create_best_ckpt_exporter(
                    model_dir,
                    params.trainer.best_ema_checkpoint_export_subdir,
                    params.trainer.best_ema_checkpoint_eval_metric,
                    params.trainer.best_ema_checkpoint_metric_comp)

                ema_trainer = None
                # If we use gradient accumulation or EMA, we need to use our EmaTrainer.
                if params.trainer.accumulation_steps > 1 or params.trainer.optimizer_config.ema is not None:
                    gradient_accumulator = GradientAccumulator()
                    ema_trainer = network_trainer.EmaTrainer(
                        params,
                        task,
                        model=model,
                        optimizer=optimizer,
                        train='train' in FLAGS.mode,
                        evaluate='eval' in FLAGS.mode,
                        checkpoint_exporter=best_ckpt_exporter,
                        ema_checkpoint_exporter=best_ema_ckpt_exporter,
                        gradient_accumulator=gradient_accumulator,
                        model_dir=model_dir,
                    )

            preemption_watcher = tf.distribute.experimental.PreemptionWatcher()

            train_lib.run_experiment(distribution_strategy=distribution_strategy,
                                     task=task,
                                     mode=FLAGS.mode,
                                     params=params,
                                     model_dir=model_dir,
                                     trainer=ema_trainer,
                                     summary_manager=None,
                                     eval_summary_manager=summary_manager.maybe_build_eval_summary_manager(
                                         params=params, model_dir=model_dir), )

            keep_training = False
        except tf.errors.OpError as e:
            if preemption_watcher and preemption_watcher.preemption_message:
                preemption_watcher.block_until_worker_exit()
                logging.info('Some TPU workers had been preempted (message: %s), '
                             'retarting training from the last checkpoint...', preemption_watcher.preemption_message)
                keep_training = True
            else:
                raise e from None


def main(_):
    logging.get_absl_handler().use_absl_log_file(program_name='network')

    gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_params)
    params = train_utils.parse_configuration(FLAGS)

    # Fix the seed for reproducibility
    logging.info(f'Seed: {FLAGS.seed}')
    tf.keras.utils.set_random_seed(int(FLAGS.seed))

    model_dir = FLAGS.model_dir
    if 'train' in FLAGS.mode:
        # Pure eval modes do not output yaml files. Otherwise continuous eval job
        # may race against the train job for writing the same file.
        train_utils.serialize_config(params, model_dir)

    # Sets mixed_precision policy. Using 'mixed_float16' or 'mixed_bfloat16'
    # can have significant impact on model speeds by utilizing float16 in case of
    # GPUs, and bfloat16 in the case of TPUs. loss_scale takes effect only when
    # dtype is float16
    if params.runtime.mixed_precision_dtype:
        performance.set_mixed_precision_policy(params.runtime.mixed_precision_dtype)

    summary_model(params, model_dir)
    _run_experiment_with_preemption_recovery(params, model_dir)
    train_utils.save_gin_config(FLAGS.mode, model_dir)


if __name__ == '__main__':
    tfm_flags.define_flags()
    flags.DEFINE_string('seed', default="0", help='Fix the seed for reproducibility')
    flags.mark_flags_as_required(['experiment', 'mode', 'model_dir'])
    app.run(main)
