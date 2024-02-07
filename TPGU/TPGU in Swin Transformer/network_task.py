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

"""An example task definition for image classification."""
from typing import Any, List, Optional, Tuple, Sequence, Mapping

import tensorflow as tf
from absl import logging

import network_config as exp_cfg
import network_input
import network_model
from official.common import dataset_fn
from official.core import base_task
from official.core import task_factory
from official.modeling import tf_utils
from official.vision.dataloaders import input_reader
from official.vision.dataloaders import input_reader_factory
from official.vision.ops import augment

from gradient_accumulator import GradientAccumulator


@task_factory.register_task_cls(exp_cfg.ExampleTask)
class ExampleTask(base_task.Task):
    """Class of an example task.

    A task is a subclass of base_task.Task that defines model, input, loss, metric
    and one training and evaluation step, etc.
    """

    def build_model(self) -> tf.keras.Model:
        """Builds a model."""
        input_specs = tf.keras.layers.InputSpec(shape=[None] + self.task_config.model.input_size)

        l2_weight_decay = self.task_config.losses.l2_weight_decay
        l2_regularizer = (tf.keras.regularizers.l2(l2_weight_decay / 2.0) if (
                l2_weight_decay is not None and l2_weight_decay > 0) else None)
        model = network_model.build_model(model_config=self.task_config.model)
        return model

    def build_inputs(self, params: exp_cfg.ExampleDataConfig,
                     input_context: Optional[tf.distribute.InputContext] = None) -> tf.data.Dataset:
        """Builds input.

        The input from this function is a tf.data.Dataset that has gone through
        pre-processing steps, such as augmentation, batching, shuffling, etc.

        Args:
          params: The experiment config.
          input_context: An optional InputContext used by input reader.

        Returns:
          A tf.data.Dataset object.
        """

        num_classes = self.task_config.model.num_classes
        input_size = self.task_config.model.input_size
        image_field_key = self.task_config.train_data.image_field_key
        label_field_key = self.task_config.train_data.label_field_key
        is_multilabel = self.task_config.train_data.is_multilabel

        decoder = network_input.Decoder(image_field_key, label_field_key, is_multilabel)
        parser = network_input.Parser(output_size=input_size[:2], num_classes=num_classes,
                                      image_field_key=image_field_key, label_field_key=label_field_key,
                                      decode_jpeg_only=params.decode_jpeg_only, aug_rand_hflip=params.aug_rand_hflip,
                                      aug_crop=params.aug_crop, aug_type=params.aug_type,
                                      color_jitter=params.color_jitter, random_erasing=params.random_erasing,
                                      is_multilabel=is_multilabel, dtype=params.dtype,
                                      center_crop_fraction=params.center_crop_fraction,
                                      tf_resize_method=params.tf_resize_method, three_augment=params.three_augment)

        postprocess_fn = None
        if params.mixup_and_cutmix is not None:
            postprocess_fn = augment.MixupAndCutmix(mixup_alpha=params.mixup_and_cutmix.mixup_alpha,
                                                    cutmix_alpha=params.mixup_and_cutmix.cutmix_alpha,
                                                    prob=params.mixup_and_cutmix.prob,
                                                    label_smoothing=params.mixup_and_cutmix.label_smoothing,
                                                    num_classes=num_classes)

        def sample_fn(repeated_augment, dataset):
            weights = [1 / repeated_augment] * repeated_augment
            dataset = tf.data.Dataset.sample_from_datasets(datasets=[dataset] * repeated_augment, weights=weights,
                                                           seed=None, stop_on_empty_dataset=True, )
            return dataset

        is_repeated_augment = (params.is_training and params.repeated_augment is not None)
        reader = input_reader_factory.input_reader_generator(params,
                                                             dataset_fn=dataset_fn.pick_dataset_fn(params.file_type),
                                                             decoder_fn=decoder.decode,
                                                             combine_fn=input_reader.create_combine_fn(params),
                                                             parser_fn=parser.parse_fn(params.is_training),
                                                             postprocess_fn=postprocess_fn, sample_fn=(
                lambda ds: sample_fn(params.repeated_augment, ds)) if is_repeated_augment else None, )

        dataset = reader.read(input_context=input_context)
        return dataset

    def build_losses(self, labels: tf.Tensor, model_outputs: tf.Tensor, aux_losses: Optional[Any] = None) -> tf.Tensor:
        """Builds losses for training and validation.

        Args:
          labels: Input groundt-ruth labels.
          model_outputs: Output of the model.
          aux_losses: The auxiliarly loss tensors, i.e. `losses` in tf.keras.Model.
        Returns:
          The total loss tensor.
        """
        # 获取losses配置
        losses_config = self.task_config.losses

        if losses_config.soft_labels:
            total_loss = tf.nn.softmax_cross_entropy_with_logits(labels, model_outputs)
        elif losses_config.one_hot:
            total_loss = tf.keras.losses.categorical_crossentropy(labels, model_outputs, from_logits=True,
                                                                  label_smoothing=losses_config.label_smoothing)
        else:
            total_loss = tf.keras.losses.sparse_categorical_crossentropy(labels, model_outputs, from_logits=True)

        total_loss = tf_utils.safe_mean(total_loss)
        if aux_losses:
            total_loss += tf.add_n(aux_losses)

        return total_loss


    def build_metrics(self, training: bool = True, prefix="") -> Sequence[tf.keras.metrics.Metric]:
        """Gets streaming metrics for training/validation.

        This function builds and returns a list of metrics to compute during
        training and validation. The list contains objects of subclasses of
        tf.keras.metrics.Metric. Training and validation can have different metrics.

        Args:
          training: Whether the metric is for training or not.

        Returns:
         A list of tf.keras.metrics.Metric objects.
        """
        if prefix != "" and (not prefix.endswith("_")):
            prefix = prefix + "_"

        logging.info(f'prefix for metric: "{prefix}"')
        k = self.task_config.evaluation.top_k
        if (self.task_config.losses.one_hot or self.task_config.losses.soft_labels):
            metrics = [tf.keras.metrics.CategoricalAccuracy(name=f'{prefix}accuracy'),
                       tf.keras.metrics.TopKCategoricalAccuracy(k=k, name=f'{prefix}top_{k}_accuracy')]
            if hasattr(self.task_config.evaluation, 'precision_and_recall_thresholds') and \
                    self.task_config.evaluation.precision_and_recall_thresholds:
                thresholds = self.task_config.evaluation.precision_and_recall_thresholds  # pylint: disable=line-too-long
                # pylint:disable=g-complex-comprehension
                metrics += [
                    tf.keras.metrics.Precision(thresholds=th, name=f'{prefix}precision_at_threshold_{th}', top_k=1) for
                    th in thresholds]
                metrics += [tf.keras.metrics.Recall(thresholds=th, name=f'{prefix}recall_at_threshold_{th}', top_k=1)
                            for th in thresholds]

                # Add per-class precision and recall.
                if hasattr(self.task_config.evaluation, 'report_per_class_precision_and_recall') and \
                        self.task_config.evaluation.report_per_class_precision_and_recall:
                    for class_id in range(self.task_config.model.num_classes):
                        metrics += [tf.keras.metrics.Precision(thresholds=th, class_id=class_id,
                                                               name=f'{prefix}precision_at_threshold_{th}/{class_id}',
                                                               top_k=1) for th in thresholds]
                        metrics += [tf.keras.metrics.Recall(thresholds=th, class_id=class_id,
                                                            name=f'{prefix}recall_at_threshold_{th}/{class_id}',
                                                            top_k=1) for th in
                                    thresholds]  # pylint:enable=g-complex-comprehension
        else:
            metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name=f'{prefix}accuracy'),
                       tf.keras.metrics.SparseTopKCategoricalAccuracy(k=k, name=f'{prefix}top_{k}_accuracy')]

        return metrics

    def train_step(self, inputs: Tuple[Any, Any], model: tf.keras.Model, optimizer: tf.keras.optimizers.Optimizer,
                   metrics: Optional[List[Any]] = None) -> Mapping[str, Any]:
        """Does forward and backward.

        This example assumes input is a tuple of (features, labels), which follows
        the output from data loader, i.e., Parser. The output from Parser is fed
        into train_step to perform one step forward and backward pass. Other data
        structure, such as dictionary, can also be used, as long as it is consistent
        between output from Parser and input used here.

        Args:
          inputs: A tuple of input tensors of (features, labels).
          model: A tf.keras.Model instance.
          optimizer: The optimizer for this training step.
          metrics: A nested structure of metrics objects.

        Returns:
          A dictionary of logs.
        """
        features, labels = inputs
        num_replicas = tf.distribute.get_strategy().num_replicas_in_sync
        with tf.GradientTape() as tape:
            outputs = model(features, training=True)
            # Casting output layer as float32 is necessary when mixed_precision is
            # mixed_float16 or mixed_bfloat16 to ensure output is casted as float32.
            outputs = tf.nest.map_structure(lambda x: tf.cast(x, tf.float32), outputs)

            # Computes per-replica loss.
            loss = self.build_losses(model_outputs=outputs, labels=labels, aux_losses=model.losses)
            # Scales loss as the default gradients allreduce performs sum inside the
            # optimizer.
            scaled_loss = loss / num_replicas

            # For mixed_precision policy, when LossScaleOptimizer is used, loss is
            # scaled for numerical stability.
            if isinstance(optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
                scaled_loss = optimizer.get_scaled_loss(scaled_loss)

        tvars = model.trainable_variables
        grads = tape.gradient(scaled_loss, tvars)
        # Scales back gradient before apply_gradients when LossScaleOptimizer is
        # used.
        if isinstance(optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
            grads = optimizer.get_unscaled_gradients(grads)
        optimizer.apply_gradients(list(zip(grads, tvars)))

        logs = {self.loss: loss}
        if metrics:
            self.process_metrics(metrics, labels, outputs)
        return logs

    def train_step_with_gradient_accumulation(self,
                                              inputs: Tuple[Any, Any],
                                              model: tf.keras.Model,
                                              optimizer: tf.keras.optimizers.Optimizer,
                                              gradient_accumulator: GradientAccumulator,
                                              can_apply_gradients: bool,
                                              gradient_accumulation_steps: int = 1,
                                              metrics: Optional[List[Any]] = None) -> Mapping[str, Any]:
        """Does forward and backward with gradient_accumulation"""
        features, labels = inputs
        num_replicas = tf.distribute.get_strategy().num_replicas_in_sync

        with tf.GradientTape() as tape:
            outputs = model(features, training=True)
            # Casting output layer as float32 is necessary when mixed_precision is
            # mixed_float16 or mixed_bfloat16 to ensure output is casted as float32.
            outputs = tf.nest.map_structure(lambda x: tf.cast(x, tf.float32), outputs)

            # Computes per-replica loss.
            loss = self.build_losses(model_outputs=outputs, labels=labels, aux_losses=model.losses)
            # Scales loss as the default gradients allreduce performs sum inside the
            # optimizer.
            scaled_loss = loss / num_replicas

            # For mixed_precision policy, when LossScaleOptimizer is used, loss is
            # scaled for numerical stability.
            if isinstance(optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
                scaled_loss = optimizer.get_scaled_loss(scaled_loss)

        tvars = model.trainable_variables
        grads = tape.gradient(scaled_loss, tvars)
        # Scales back gradient before apply_gradients when LossScaleOptimizer is used.
        if isinstance(optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
            grads = optimizer.get_unscaled_gradients(grads)

        # Accumulate gradients
        gradient_accumulator(gradients=grads)

        if can_apply_gradients:
            gradients = [grad / gradient_accumulation_steps for grad in gradient_accumulator.gradients]
            optimizer.apply_gradients(list(zip(gradients, tvars)))
            gradient_accumulator.reset()

        logs = {self.loss: loss}
        if metrics:
            self.process_metrics(metrics, labels, outputs)
        return logs

    def validation_step(self, inputs: Tuple[Any, Any],
                        model: tf.keras.Model,
                        metrics: Optional[List[Any]] = None) -> Mapping[str, Any]:
        """Runs validation step.

        Args:
          inputs: A tuple of input tensors of (features, labels).
          model: A tf.keras.Model instance.
          metrics: A nested structure of metrics objects.

        Returns:
          A dictionary of logs.
        """
        features, labels = inputs
        one_hot = self.task_config.losses.one_hot
        soft_labels = self.task_config.losses.soft_labels
        is_multilabel = self.task_config.train_data.is_multilabel
        # Note: `soft_labels`` only apply to the training phrase. In the validation
        # phrase, labels should still be integer ids and need to be converted to
        # one hot format.
        if (one_hot or soft_labels) and not is_multilabel:
            labels = tf.one_hot(labels, self.task_config.model.num_classes)

        outputs = self.inference_step(features, model)
        outputs = tf.nest.map_structure(lambda x: tf.cast(x, tf.float32), outputs)
        loss = self.build_losses(model_outputs=outputs, labels=labels, aux_losses=model.losses)

        logs = {self.loss: loss}
        # Convert logits to softmax for metric computation if needed.
        if hasattr(self.task_config.model, 'output_softmax') and self.task_config.model.output_softmax:
            outputs = tf.nn.softmax(outputs, axis=-1)
        if metrics:
            self.process_metrics(metrics, labels, outputs)
        elif model.compiled_metrics:
            self.process_compiled_metrics(model.compiled_metrics, labels, outputs)
            logs.update({m.name: m.result() for m in model.metrics})
        return logs

    def inference_step(self, inputs: tf.Tensor, model: tf.keras.Model) -> Any:
        """Performs the forward step. It is used in 'validation_step'."""
        return model(inputs, training=False)
