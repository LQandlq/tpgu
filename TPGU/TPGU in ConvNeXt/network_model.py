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

import logging
import math
from typing import Any, Mapping

import tensorflow as tf
from numpy import linspace

import network_config
from official.vision.modeling.layers.nn_layers import StochasticDepth

keras_layers = tf.keras.layers
keras_initializers = tf.keras.initializers
keras_model = tf.keras.Model


class DownSample(keras_layers.Layer):
    def __init__(self, dim, kernel_initializer=None, kernel_regularizer=None):
        super().__init__()
        self.norm = keras_layers.LayerNormalization(epsilon=1e-6)
        self.conv = keras_layers.Conv2D(filters=dim,
                                        kernel_size=2,
                                        strides=2,
                                        kernel_initializer=kernel_initializer,
                                        kernel_regularizer=kernel_regularizer)

    def call(self, x):
        x = self.norm(x)
        x = self.conv(x)
        return x


class TaylorPolynomial(keras_layers.Layer):
    def __init__(self, order, dim, kernel_initializer=None):
        super().__init__()
        self.order = order
        self.a = [self.add_weight(shape=[dim],
                                  initializer=kernel_initializer if i != 0 else keras_initializers.Constant(1.0),
                                  trainable=True,
                                  name=f'a{i}_for_order{i}')
                  for i in range(order + 1)]
        self.b = self.add_weight(shape=[dim],
                                 initializer=keras_initializers.Constant(0.0),
                                 trainable=True,
                                 name='polynomial_bias')
        self.factorial = [1 / math.factorial(i) for i in range(order + 1)]
        self.sigmoid = keras_layers.Activation('sigmoid')

    def call(self, x):
        f = self.sigmoid(1.702 * self.b)
        p = f * (1 - f)
        q = 1 - 2 * f

        x = x - self.b
        power, polynomial = 1, 0
        for order in range(self.order + 1):
            derivative = self.get_derivative(f, p, q, order)
            polynomial = self.a[order] * self.factorial[order] * derivative * power + polynomial
            power = power * x
        return polynomial

    def get_derivative(self, f, p, q, order):
        alpha = 1.702 ** order
        if order == 0:
            return f
        if order == 1:
            return alpha * p
        if order == 2:
            return alpha * p * q
        if order == 3:
            return alpha * p * (1 - 6 * p)
        if order == 4:
            return alpha * p * q * (1 - 12 * p)
        if order == 5:
            return alpha * p * (1 - 30 * p + 120 * p * p)
        if order == 6:
            return alpha * p * q * (1 - 60 * p + 360 * p * p)
        if order == 7:
            return alpha * p * (1 - 126 * p + 1680 * p * p - 5040 * (p ** 3))
        if order == 8:
            return alpha * p * q * (1 - 252 * p + 5040 * p * p - 20160 * (p ** 3))


class Block(keras_layers.Layer):
    def __init__(self, dim, order, drop_rate=0.1, layer_scale_init_value=1e-6, kernel_initializer=None,
                 kernel_regularizer=None):
        """
            Args:
                dim (int): Number of input channels.
                drop_rate (float): Stochastic depth rate. Default: 0.1
                layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        """
        super().__init__()

        self.dwconv = keras_layers.DepthwiseConv2D(kernel_size=7,
                                                   padding='same',
                                                   kernel_initializer=kernel_initializer,
                                                   kernel_regularizer=kernel_regularizer)
        self.norm = keras_layers.LayerNormalization(epsilon=1e-6)
        self.pwconv1 = keras_layers.Dense(units=dim * 4,
                                          kernel_initializer=kernel_initializer,
                                          kernel_regularizer=kernel_regularizer)
        self.taylor_polynomial = TaylorPolynomial(order, dim * 4, kernel_initializer)
        self.pwconv2 = keras_layers.Dense(units=dim,
                                          kernel_initializer=kernel_initializer,
                                          kernel_regularizer=kernel_regularizer)

        if layer_scale_init_value > 0:
            # warning: cannot save a model without giving a name in Layer.add_weight call
            self.gamma = self.add_weight(shape=(dim,),
                                         initializer=keras_initializers.Constant(layer_scale_init_value),
                                         trainable=True,
                                         name='layer_scale_gamma')
        else:
            self.gamma = 1

        self.drop_path_enabled = True if drop_rate > 0 else False
        self.drop_path = StochasticDepth(drop_rate) if drop_rate > 0 else None

    def call(self, x, training=None):
        shortcut = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.taylor_polynomial(x) * x
        x = self.pwconv2(x)
        x = self.gamma * x
        x = self.drop_path(x, training) if self.drop_path_enabled else x
        x = shortcut + x
        return x


class Network(keras_model):
    """An example model class.

    A model is a subclass of tf.keras.Model where layers are built in the
    constructor.
    """

    def __init__(self,
                 num_classes,
                 dims,
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
        logging.info(f'dp_rates={dp_rates}')

        orders = [7, 7, 7, 7]
        logging.info(f'orders={orders}')
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

        self.config_dict = {
            'num_classes': num_classes,
            'dims': dims,
            'drop_path_rate': drop_path_rate,
            'depths': depths,
            'kernel_initializer': kernel_initializer,
            'kernel_regularizer': kernel_regularizer,
            'input_specs': input_specs
        }

    def get_config(self) -> Mapping[str, Any]:
        """Returns the config dictionary for recreating this class."""
        return self.config_dict

    @classmethod
    def from_config(cls, config, custom_objects=None):
        """Constructs an instance of this model from input config."""
        return cls(**config)


def build_model(input_specs: keras_layers.InputSpec,
                model_config: network_config.ModelConfig,
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

    return Network(num_classes=model_config.num_classes,
                   dims=model_config.dims,
                   drop_path_rate=model_config.drop_path_rate,
                   depths=model_config.depths,
                   kernel_initializer=model_config.kernel_initializer,
                   kernel_regularizer=kernel_regularizer,
                   input_specs=input_specs, **kwargs)
