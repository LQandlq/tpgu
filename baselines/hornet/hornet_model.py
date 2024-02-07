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

"""A sample model implementation.

This is only a dummy example to showcase how a model is composed. It is usually
not needed to implement a model from scratch. Most SoTA models can be found and
directly used from `official/vision/modeling` directory.
"""

from typing import Any, Mapping

from numpy import linspace
import tensorflow as tf
from official.vision.modeling.layers.nn_layers import StochasticDepth

import hornet_config as model_cfg

keras_layers = tf.keras.layers
keras_initializers = tf.keras.initializers
keras_model = tf.keras.Model


class DownSample(keras_layers.Layer):
    def __init__(self, dim, kernel_initializer=None, kernel_regularizer=None):
        super().__init__()
        self._norm = keras_layers.LayerNormalization(epsilon=1e-6)
        self._conv = keras_layers.Conv2D(filters=dim,
                                         kernel_size=2,
                                         strides=2,
                                         kernel_initializer=kernel_initializer,
                                         kernel_regularizer=kernel_regularizer)

    def call(self, x):
        x = self._norm(x)
        x = self._conv(x)
        return x


# gnconv
class GnConv(keras_layers.Layer):
    def __init__(self, dim, order=5, kernel_initializer=None, kernel_regularizer=None):
        super().__init__()
        self._order = order
        self._dims = [dim // 2 ** i for i in range(order)]
        self._dims.reverse()

        self._proj_in = keras_layers.Dense(units=dim * 2,
                                           kernel_initializer=kernel_initializer,
                                           kernel_regularizer=kernel_regularizer)

        self._dwconv = keras_layers.DepthwiseConv2D(kernel_size=7, padding="same",
                                                    depthwise_initializer=kernel_initializer,
                                                    depthwise_regularizer=kernel_regularizer)

        self._proj_out = keras_layers.Dense(units=dim,
                                            kernel_initializer=kernel_initializer,
                                            kernel_regularizer=kernel_regularizer)

        self._pws = [keras_layers.Dense(units=self._dims[i + 1],
                                        kernel_initializer=kernel_initializer,
                                        kernel_regularizer=kernel_regularizer)
                     for i in range(order - 1)]

        self._scale = 1.0 / 3.0

    def call(self, x):
        fused_x = self._proj_in(x)
        pwa, abc = tf.split(fused_x, (self._dims[0], sum(self._dims)), axis=3)

        dw_abc = self._dwconv(abc) * self._scale
        dw_list = tf.split(dw_abc, self._dims, axis=3)

        x = pwa * dw_list[0]
        for i in range(self._order - 1):
            x = self._pws[i](x) * dw_list[i + 1]

        x = self._proj_out(x)
        return x


class Block(keras_layers.Layer):
    def __init__(self, order, dim, drop_rate=0.1, layer_scale_init_value=1e-6, kernel_initializer=None,
                 kernel_regularizer=None):
        """
            Args:
                dim (int): Number of input channels.
                drop_rate (float): Stochastic depth rate. Default: 0.0
                layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        """
        super().__init__()

        self._norm1 = keras_layers.LayerNormalization(epsilon=1e-6)
        self._gn_Conv = GnConv(dim, order, kernel_initializer, kernel_regularizer)

        self._norm2 = keras_layers.LayerNormalization(epsilon=1e-6)
        self._pwconv1 = keras_layers.Dense(units=4 * dim,
                                           kernel_initializer=kernel_initializer,
                                           kernel_regularizer=kernel_regularizer)
        self._act = keras_layers.Activation("gelu")
        self._pwconv2 = keras_layers.Dense(units=dim,
                                           kernel_initializer=kernel_initializer,
                                           kernel_regularizer=kernel_regularizer)

        if layer_scale_init_value > 0:
            # warning: cannot save a model without giving a name in Layer.add_weight call
            self._gamma1 = self.add_weight(shape=[dim],
                                           initializer=keras_initializers.Constant(layer_scale_init_value),
                                           trainable=True,
                                           name='layer_scale1_gamma')
            self._gamma2 = self.add_weight(shape=[dim],
                                           initializer=keras_initializers.Constant(layer_scale_init_value),
                                           trainable=True,
                                           name='layer_scale2_gamma')
        else:
            self._gamma1 = 1
            self._gamma2 = 1

        self._drop_path_enabled = True if drop_rate > 0 else False
        self._drop_path = StochasticDepth(drop_rate) if drop_rate > 0 else None

    def call(self, x, training=None):
        shortcut = x
        x = self._norm1(x)
        x = self._gn_Conv(x)
        x = self._gamma1 * x
        x = self._drop_path(x, training) if self._drop_path_enabled else x
        x = shortcut + x

        shortcut = x
        x = self._norm2(x)
        x = self._pwconv1(x)
        x = self._act(x)
        x = self._pwconv2(x)
        x = self._gamma2 * x
        x = self._drop_path(x, training) if self._drop_path_enabled else x
        x = shortcut + x
        return x


class Horent(keras_model):
    """An example model class.

    A model is a subclass of tf.keras.Model where layers are built in the
    constructor.
    """

    def __init__(self,
                 num_classes,
                 dims,
                 orders,
                 drop_path_rate,
                 depths=[3, 3, 9, 3],
                 kernel_initializer="truncated_normal",
                 kernel_regularizer=None,
                 input_specs=keras_layers.InputSpec(shape=[None, None, None, 3]),
                 **kwargs):
        """Initializes the example model.

        All layers are defined in the constructor, and config is recorded in the
        `_config_dict` object for serialization.

        Args:
          num_classes: The number of classes in classification task.
          input_specs: A `tf.keras.layers.InputSpec` spec of the input tensor.
          **kwargs: Additional keyword arguments to be passed.
        """

        inputs = tf.keras.Input(shape=input_specs.shape[1:], name=input_specs.name)

        if kernel_initializer == "truncated_normal":
            kernel_initializer = dict(class_name='TruncatedNormal', config=dict(stddev=.02))

        # stem
        x = keras_layers.Conv2D(
            filters=dims[0],
            kernel_size=4,
            strides=4,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
        )(inputs)
        x = keras_layers.LayerNormalization(epsilon=1e-6)(x)

        # 根据深度产生对应的drop rate
        cur = 0
        dp_rates = linspace(start=0, stop=drop_path_rate, num=sum(depths))

        # stages
        for i, depth in enumerate(depths):
            for _ in range(depth):
                x = Block(dim=dims[i],
                          order=orders[i],
                          drop_rate=dp_rates[cur],
                          kernel_initializer=kernel_initializer,
                          kernel_regularizer=kernel_regularizer)(x)
                cur += 1
            if i != len(depths) - 1:
                x = DownSample(dims[i + 1],
                               kernel_initializer=kernel_initializer,
                               kernel_regularizer=kernel_regularizer)(x)

        # head
        x = tf.reduce_mean(x, axis=[1, 2])
        x = keras_layers.LayerNormalization(epsilon=1e-6)(x)
        outputs = keras_layers.Dense(units=num_classes,
                                     kernel_initializer=kernel_initializer,
                                     kernel_regularizer=kernel_regularizer)(x)

        super().__init__(inputs=inputs, outputs=outputs, **kwargs)

        self._config_dict = {
            'num_classes': num_classes,
            'dims': dims,
            'orders': orders,
            'drop_path_rate': drop_path_rate,
            'depths': depths,
            'kernel_initializer': kernel_initializer,
            'kernel_regularizer': kernel_regularizer,
            'input_specs': input_specs
        }

    def get_config(self) -> Mapping[str, Any]:
        """Returns the config dictionary for recreating this class."""
        return self._config_dict

    @classmethod
    def from_config(cls, config, custom_objects=None):
        """Constructs an instance of this model from input config."""
        return cls(**config)


def build_model(input_specs: keras_layers.InputSpec,
                model_config: model_cfg.ExampleModel,
                kernel_regularizer=None,
                **kwargs) -> keras_model:
    """Builds and returns the example model.

    This function is the main entry point to build a model. Commonly, it builds a
    model by building a backbone, decoder and head. An example of building a
    classification model is at
    third_party/tensorflow_models/official/vision/modeling/backbones/resnet.py.
    However, it is not mandatory for all models to have these three pieces
    exactly. Depending on the task, model can be as simple as the example model
    here or more complex, such as multi-head architecture.

    Args:
      input_specs: The specs of the input layer that defines input size.
      model_config: The config containing parameters to build a model.
      **kwargs: Additional keyword arguments to be passed.

    Returns:
      A tf.keras.Model object.
    """

    return Horent(num_classes=model_config.num_classes,
                  dims=model_config.dims,
                  orders=model_config.orders,
                  drop_path_rate=model_config.drop_path_rate,
                  depths=model_config.depths,
                  kernel_initializer=model_config.kernel_initializer,
                  kernel_regularizer=kernel_regularizer,
                  input_specs=input_specs, **kwargs)
