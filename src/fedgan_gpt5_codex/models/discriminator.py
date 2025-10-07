"""Discriminator model for the FedGAN prototype.

Implements the DCGAN discriminator that maps an MNIST image to a single logit.
"""
from __future__ import annotations

from typing import Tuple
import tensorflow as tf
from tensorflow import keras  # type: ignore[attr-defined]

_IMAGE_SHAPE: Tuple[int, int, int] = (28, 28, 1)


def build_discriminator() -> keras.Sequential:
    """Create a discriminator network following the DCGAN tutorial setup."""
    model = keras.Sequential(name="discriminator")
    model.add(
        keras.layers.Conv2D(
            64,
            kernel_size=(5, 5),
            strides=(2, 2),
            padding="same",
            input_shape=_IMAGE_SHAPE,
        )
    )
    model.add(keras.layers.LeakyReLU())
    model.add(keras.layers.Dropout(0.3))

    model.add(
        keras.layers.Conv2D(
            128,
            kernel_size=(5, 5),
            strides=(2, 2),
            padding="same",
        )
    )
    model.add(keras.layers.LeakyReLU())
    model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(1))
    model.build((None, *_IMAGE_SHAPE))
    return model


def discriminator_loss(real_output: tf.Tensor, fake_output: tf.Tensor) -> tf.Tensor:
    """Standard DCGAN discriminator loss using binary cross-entropy."""
    cross_entropy = keras.losses.BinaryCrossentropy(from_logits=True)
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss
