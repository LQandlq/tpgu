# -*- coding: utf-8 -*-
"""
Created on 2023/7/11

@author: EssenceOfTheWorld

This class is simplified from GradientAccumulator (https://github.com/OpenNMT/OpenNMT-tf/opennmt/optimizers/utils.py)
"""

import tensorflow as tf


class GradientAccumulator:
    """Gradient accumulation utility.

    When used with a distribution strategy, the accumulator should be called in a
    replica context. Gradients will be accumulated locally on each replica and
    without synchronization. Users should then call ``.gradients``, scale the
    gradients if required, and pass the result to ``apply_gradients``.
    """

    # We use the ON_READ synchronization policy so that no synchronization is
    # performed on assignment. To get the value, we call .value() which returns the
    # value on the current replica without synchronization.

    def __init__(self):
        """Initializes the accumulator."""
        self._gradients = []

    @property
    def gradients(self):
        """The accumulated gradients on the current replica."""
        if not self._gradients:
            raise ValueError(
                "The accumulator should be called first to initialize the gradients"
            )
        return list(gradient.value() for gradient in self._gradients)

    def __call__(self, gradients):
        """Accumulates :obj:`gradients` on the current replica."""
        if not self._gradients:
            self._gradients.extend(
                [
                    tf.Variable(
                        tf.zeros_like(gradient),
                        trainable=False,
                        synchronization=tf.VariableSynchronization.ON_READ,
                    )
                    for gradient in gradients
                ]
            )
        if len(gradients) != len(self._gradients):
            raise ValueError(
                "Expected %s gradients, but got %d"
                % (len(self._gradients), len(gradients))
            )

        for accum_gradient, gradient in zip(self._gradients, gradients):
            accum_gradient.assign_add(gradient, read_value=False)

    def reset(self):
        """Resets the accumulated gradients on the current replica."""
        if not self._gradients:
            return
        for gradient in self._gradients:
            shape = (
                gradient.shape
                if gradient.shape.is_fully_defined()
                else tf.shape(gradient)
            )
            gradient.assign(tf.zeros(shape, dtype=gradient.dtype), read_value=False)
