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


import dataclasses
from typing import List, Optional, Tuple, Union, Sequence

from official.core import config_definitions as cfg
from official.core import exp_factory
from official.modeling import hyperparams
from official.modeling import optimization
from official.vision.configs import common


@dataclasses.dataclass
class DataConfig(cfg.DataConfig):
    """Input config for training. Add more fields as needed."""
    input_path: Union[Sequence[str], str, hyperparams.Config] = ''
    weights: Optional[hyperparams.base_config.Config] = None
    global_batch_size: int = 0
    is_training: bool = True
    dtype: str = 'bfloat16'
    shuffle_buffer_size: int = 10000
    # cycle_length: int = 10
    is_multilabel: bool = False
    aug_rand_hflip: bool = True
    aug_crop: Optional[bool] = True
    crop_area_range: Optional[Tuple[float, float]] = (0.08, 1.0)
    aug_type: Optional[common.Augmentation] = None  # Choose from AutoAugment and RandAugment.
    three_augment: bool = False
    color_jitter: float = 0.
    random_erasing: Optional[common.RandomErasing] = None
    file_type: str = 'tfrecord'
    image_field_key: str = 'image/encoded'
    label_field_key: str = 'image/class/label'
    decode_jpeg_only: bool = True
    mixup_and_cutmix: Optional[common.MixupAndCutmix] = None

    # Keep for backward compatibility: None, 'autoaug', or 'randaug'.
    aug_policy: Optional[str] = 'randaug'
    randaug_magnitude: Optional[int] = 10
    # Determines ratio between the side of the cropped image and the short side of the original image.
    center_crop_fraction: Optional[float] = 0.875
    # Interpolation method for resizing image in Parser for both training and eval
    tf_resize_method: str = 'bicubic'
    # Repeat augmentation puts multiple augmentations of the same image in a batch https://arxiv.org/abs/1902.05509
    repeated_augment: Optional[int] = None


@dataclasses.dataclass
class ModelConfig(hyperparams.Config):
    """The model config. Used by build_model function."""
    num_classes: int = 0
    dims: List[int] = dataclasses.field(default_factory=list)
    orders: List[int] = dataclasses.field(default_factory=list)
    drop_path_rate: float = 0.1
    depths: List[int] = dataclasses.field(default_factory=list)
    kernel_initializer: str = 'truncated_normal'
    input_size: List[int] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class LossesConfig(hyperparams.Config):
    l2_weight_decay: float = 0.0
    label_smoothing: float = 0.1
    one_hot: bool = False
    from_logits: bool = False
    soft_labels: bool = False


@dataclasses.dataclass
class EvaluationConfig(hyperparams.Config):
    top_k: int = 5
    precision_and_recall_thresholds: Optional[List[float]] = None
    report_per_class_precision_and_recall: bool = False


@dataclasses.dataclass
class TrainerConfig(cfg.TrainerConfig):
    """Configuration for trainer. """
    accumulation_steps: int = 1
    best_ema_checkpoint_export_subdir: str = "ema_best_ckpt"
    best_ema_checkpoint_eval_metric: str = "ema_accuracy"
    best_ema_checkpoint_metric_comp: str = "higher"
    max_to_keep = 2


@dataclasses.dataclass
class TaskConfig(cfg.TaskConfig):
    """The task config."""
    model: ModelConfig = ModelConfig()
    train_data: DataConfig = DataConfig(is_training=True)
    validation_data: DataConfig = DataConfig(is_training=False)
    losses: LossesConfig = LossesConfig()
    evaluation: EvaluationConfig = EvaluationConfig()


@exp_factory.register_config_factory('vision_experiment')
def vision_experiment() -> cfg.ExperimentConfig:
    """Definition of a full example experiment."""
    steps_per_epoch = 312
    config = cfg.ExperimentConfig(task=TaskConfig(model=ModelConfig(num_classes=10, input_size=[128, 128, 3]),
                                                  losses=LossesConfig(l2_weight_decay=1e-4, label_smoothing=0.1,
                                                                      one_hot=False, from_logits=False, soft_labels=True),
                                                  train_data=DataConfig(input_path='/path/to/train*',
                                                                         is_training=True,
                                                                         global_batch_size=4096),
                                                  validation_data=DataConfig(input_path='/path/to/valid*',
                                                                              is_training=False,
                                                                              global_batch_size=4096)),
                                  trainer=TrainerConfig(steps_per_loop=steps_per_epoch,
                                                        summary_interval=steps_per_epoch,
                                                        checkpoint_interval=steps_per_epoch,
                                                        train_steps=300 * steps_per_epoch,
                                                        validation_steps=steps_per_epoch,
                                                        validation_interval=steps_per_epoch,
                                                        optimizer_config=optimization.OptimizationConfig(
                                                            {'ema': {'average_decay': 0.9999},
                                                             'optimizer': {'type': 'adamw',
                                                                           'adamw': {'weight_decay_rate': 0.05,
                                                                                     'include_in_weight_decay':
                                                                                         ['.*(kernel|weight):0$'],
                                                                                     'exclude_from_weight_decay':
                                                                                         ['bias|beta|gamma):0$'],
                                                                                     'epsilon': 1e-8,
                                                                                     }
                                                                           },
                                                             'learning_rate': {'type': 'cosine',
                                                                               'cosine': {
                                                                                   'initial_learning_rate': 0.004,
                                                                                   'decay_steps': 93600}},
                                                             'warmup': {'type': 'linear',
                                                                        'linear': {
                                                                            'warmup_steps': 20 * steps_per_epoch,
                                                                            'warmup_learning_rate': 0}}})),
                                  restrictions=['task.train_data.is_training != None',
                                                'task.validation_data.is_training != None'])

    return config
