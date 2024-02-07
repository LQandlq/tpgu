# -*- coding: utf-8 -*-
"""
Created on 2023/7/6

@author: EssenceOfTheWorld
"""

from typing import Union, Optional
from pathlib import PurePath

import gin
import tensorflow as tf
from absl import logging

import orbit
from official.core import base_task
from official.core import base_trainer
from official.core import config_definitions
from official.modeling import optimization
from orbit.utils import loop_fns

ExperimentConfig = config_definitions.ExperimentConfig
TrainerConfig = config_definitions.TrainerConfig


@gin.configurable
class EmaTrainer(base_trainer.Trainer):
    """Implements the common trainer shared for TensorFlow models."""

    def __init__(self, config: ExperimentConfig, task: base_task.Task, model: tf.keras.Model,
                 optimizer: tf.optimizers.Optimizer, train: bool = True, evaluate: bool = True,
                 train_dataset: Optional[Union[tf.data.Dataset, tf.distribute.DistributedDataset]] = None,
                 validation_dataset: Optional[Union[tf.data.Dataset, tf.distribute.DistributedDataset]] = None,
                 checkpoint_exporter=None, ema_checkpoint_exporter=None, gradient_accumulator=None, model_dir=None):
        super().__init__(config, task, model, optimizer, train, evaluate, train_dataset, validation_dataset,
                         checkpoint_exporter)

        logging.info('### Initializing ema trainer ! ###')
        self._gradient_accumulation_steps = config.trainer.accumulation_steps
        self._gradient_accumulator = gradient_accumulator
        self._ema_checkpoint_exporter = ema_checkpoint_exporter
        self._model_dir = model_dir
        if evaluate:
            logging.info("Creating a new set of validation variables for ema evaluation!")
            self._ema_validation_metrics = self.task.build_metrics(training=False, prefix="ema")
            self._ema_validation_loss = tf.keras.metrics.Mean("ema_validation_loss", dtype=tf.float32)
            self._ema_eval_loop_fn = None

    def evaluate(self, num_steps: tf.Tensor) -> Optional[orbit.runner.Output]:
        """Implements `num_steps` steps of evaluation.

        Args:
          num_steps: The number of evaluation steps to run. When this is -1,
            evaluation proceeds until a call to `eval_step` raises a `StopIteration`
            or `tf.errors.OutOfRangeError`.

        Returns:
          The output of `self.eval_end()`.

        Raises:
          ValueError: If `options.use_tf_while_loop` is `True` and `num_steps` is
            unspecified.
        """
        if self._eval_options.use_tf_while_loop and num_steps == -1:
            raise ValueError("Looping until exhausted is not supported if "
                             "`options.use_tf_while_loop` is `True`")

        # run eval
        outputs = self.eval_begin()  # pylint: disable=assignment-from-no-return

        has_state = outputs is not None
        if self._eval_loop_fn is None:
            self._eval_loop_fn = self.create_eval_loop_fn(has_state)

        eval_iter = tf.nest.map_structure(iter, self.eval_dataset)

        if self._eval_options.use_tf_while_loop and not has_state:
            self._eval_loop_fn(eval_iter, num_steps)
        else:
            outputs = self._eval_loop_fn(eval_iter, num_steps, state=outputs, reduce_fn=self.eval_reduce)

        if outputs is None:
            outputs = self.eval_end()
        else:
            outputs = self.eval_end(outputs)

        # run ema model eval
        ema_outputs = self.ema_eval_begin()  # pylint: disable=assignment-from-no-return

        has_state = ema_outputs is not None
        if self._ema_eval_loop_fn is None:
            self._ema_eval_loop_fn = self.create_ema_eval_loop_fn(has_state)

        ema_eval_iter = tf.nest.map_structure(iter, self.eval_dataset)

        if self._eval_options.use_tf_while_loop and not has_state:
            self._ema_eval_loop_fn(ema_eval_iter, num_steps)
        else:
            ema_outputs = self._ema_eval_loop_fn(ema_eval_iter, num_steps, state=ema_outputs,
                                                 reduce_fn=self.eval_reduce)

        if ema_outputs is None:
            ema_outputs = self.ema_eval_end()
        else:
            ema_outputs = self.ema_eval_end(ema_outputs)

        return {**ema_outputs, **outputs}

    def create_ema_eval_loop_fn(self, has_state: bool):
        """Creates a training loop from the given step function and options."""
        ema_eval_step_fn = self.ema_eval_step
        if self._eval_options.use_tf_while_loop:
            # even when it is not used inside the loop. To workaround this limitation,
            # we have to build two tf.functions for it.
            if has_state:
                ema_eval_loop_fn = loop_fns.create_tf_while_loop_fn_with_state(ema_eval_step_fn)
            else:
                ema_eval_loop_fn = loop_fns.create_tf_while_loop_fn(ema_eval_step_fn)
            ema_eval_loop_fn = tf.function(ema_eval_loop_fn)
        else:
            if self._eval_options.use_tf_function:
                ema_eval_step_fn = tf.function(ema_eval_step_fn)
            ema_eval_loop_fn = loop_fns.create_loop_fn(ema_eval_step_fn)

        if getattr(self, "_is_async", False):
            if has_state:
                raise ValueError("Stateful eval loop is not supported in async training.")

            def _async_ema_loop_fn(iterator, num_steps, state=None, reduce_fn=None):
                assert state is None
                assert reduce_fn is None
                self._coordinator.schedule(ema_eval_loop_fn, args=(iterator, num_steps))

            return _async_ema_loop_fn
        else:
            return ema_eval_loop_fn

    def ema_eval_step(self, iterator):
        """See base class."""

        def step_fn(inputs):
            logs = self.task.validation_step(inputs, model=self.model, metrics=self.ema_validation_metrics)
            if self.task.loss in logs:
                self._ema_validation_loss.update_state(logs[self.task.loss])
            return logs

        inputs, passthrough_logs = self.next_eval_inputs(iterator)
        distributed_outputs = self.strategy.run(step_fn, args=(inputs,))
        logs = tf.nest.map_structure(self.strategy.experimental_local_results, distributed_outputs)

        if set(logs.keys()) & set(passthrough_logs.keys()):
            logging.warning(("Conflict between the pasthrough log keys and the returned model"
                             " log keys. Found %r keys in the passthrough logs and %r keys in"
                             " the model logs. Model log keys takes precedence."), logs.keys(),
                            passthrough_logs.keys(), )

        # Python 3.8 does not support the `|` operator on dictionaries.
        # return passthrough_logs | logs
        return {**passthrough_logs, **logs}

    def ema_eval_begin(self):
        """Sets up metrics."""
        for metric in self.ema_validation_metrics + [self.ema_validation_loss]:
            metric.reset_states()
        # Swaps weights to test on weights moving average.
        if self.optimizer and isinstance(self.optimizer, optimization.ExponentialMovingAverage):
            self.optimizer.swap_weights()

    def ema_eval_end(self, aggregated_logs=None):
        """Processes evaluation results."""
        self.join()
        logs = {}
        for metric in self.ema_validation_metrics:
            logs[metric.name] = metric.result()
        if self.ema_validation_loss.count.numpy() != 0:
            logs[self.ema_validation_loss.name] = self.ema_validation_loss.result()
        else:
            # `self.validation_loss` metric was not updated, because the validation
            # loss was not returned from the task's `validation_step` method.
            logging.info("The task did not report validation loss.")
        if aggregated_logs:
            metrics = self.task.reduce_aggregated_logs(aggregated_logs, global_step=self.global_step)
            logs.update(metrics)

        if self._ema_checkpoint_exporter:
            global_step = self.global_step.numpy()
            is_new_best = self._ema_checkpoint_exporter.maybe_export_checkpoint(self.checkpoint, logs, global_step)

            metric_name = self.config.trainer.best_ema_checkpoint_eval_metric
            logs[f'best_{metric_name}'] = self._ema_checkpoint_exporter.best_ckpt_logs[metric_name]

            # if is_new_best:
            #     self._model.save(PurePath(self._model_dir, 'saved_best_ema_model'), save_format='tf')

        # Swaps back weights after testing when EMA is used.
        # This happens after best checkpoint export so that average weights used for
        # eval are exported instead of regular weights.
        if self.optimizer and isinstance(self.optimizer, optimization.ExponentialMovingAverage):
            self.optimizer.swap_weights()
        return logs

    @property
    def ema_validation_loss(self):
        """Accesses the validation loss metric object."""
        return self._ema_validation_loss

    @property
    def ema_validation_metrics(self):
        """Accesses all validation metric metric objects."""
        return self._ema_validation_metrics

    def eval_begin(self):
        """Sets up metrics."""
        for metric in self.validation_metrics + [self.validation_loss]:
            metric.reset_states()

    def eval_end(self, aggregated_logs=None):
        """Processes evaluation results."""
        self.join()
        logs = {}
        for metric in self.validation_metrics:
            logs[metric.name] = metric.result()
        if self.validation_loss.count.numpy() != 0:
            logs[self.validation_loss.name] = self.validation_loss.result()
        else:
            # `self.validation_loss` metric was not updated, because the validation
            # loss was not returned from the task's `validation_step` method.
            logging.info("The task did not report validation loss.")
        if aggregated_logs:
            metrics = self.task.reduce_aggregated_logs(aggregated_logs, global_step=self.global_step)
            logs.update(metrics)

        # best_checkpoint
        if self._checkpoint_exporter:
            global_step = self.global_step.numpy()
            is_new_best = self._checkpoint_exporter.maybe_export_checkpoint(self.checkpoint, logs, global_step)
            metric_name = self.config.trainer.best_checkpoint_eval_metric
            logs[f'best_{metric_name}'] = self._checkpoint_exporter.best_ckpt_logs[metric_name]

            # if is_new_best:
            #     self._model.save(PurePath(self._model_dir, 'saved_best_model'), save_format='tf')

        return logs

    def train_step(self, iterator):
        """See base class."""

        def step_fn(inputs, can_apply_gradients):
            if self._gradient_accumulation_steps > 1:
                logging.info('### using gradient accumulation during training ###')

                if self.config.runtime.enable_xla and (self.config.runtime.num_gpus > 0):
                    task_train_step = tf.function(self.task.train_step_with_gradient_accumulation, jit_compile=True)
                else:
                    task_train_step = self.task.train_step_with_gradient_accumulation
                logs = task_train_step(inputs,
                                       model=self.model,
                                       optimizer=self.optimizer,
                                       gradient_accumulator=self._gradient_accumulator,
                                       can_apply_gradients=can_apply_gradients,
                                       gradient_accumulation_steps=self._gradient_accumulation_steps,
                                       metrics=self.train_metrics)

            else:
                if self.config.runtime.enable_xla and (self.config.runtime.num_gpus > 0):
                    task_train_step = tf.function(self.task.train_step, jit_compile=True)
                else:
                    task_train_step = self.task.train_step
                logs = task_train_step(inputs, model=self.model, optimizer=self.optimizer, metrics=self.train_metrics)

            self._train_loss.update_state(logs[self.task.loss])
            if can_apply_gradients:
                self.global_step.assign_add(1)

        for i in range(self._gradient_accumulation_steps):
            inputs = self.next_train_inputs(iterator)
            apply_gradients = True if (i == self._gradient_accumulation_steps - 1) else False
            self.strategy.run(step_fn, args=(inputs, apply_gradients), options=self._runtime_options)

