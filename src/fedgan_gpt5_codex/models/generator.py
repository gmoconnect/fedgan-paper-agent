"""Generator model for the FedGAN prototype.

This module mirrors the DCGAN architecture from the official TensorFlow
tutorial. The generator transforms a latent noise vector into a 28×28×1 image
with values in the range [-1, 1].
"""
from __future__ import annotations

from typing import Tuple
import tensorflow as tf
from tensorflow import keras  # type: ignore[attr-defined]

_NOISE_DIM_DEFAULT = 100
_IMAGE_SHAPE: Tuple[int, int, int] = (28, 28, 1)


def build_generator(noise_dim: int = _NOISE_DIM_DEFAULT) -> keras.Sequential:
    """Create a generator network following the DCGAN tutorial setup.

    Args:
        noise_dim: Dimensionality of the latent noise vector.

    Returns:
        A ``tf.keras.Sequential`` generator model.
    """
    model = keras.Sequential(name="generator")

    model.add(
        keras.layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(noise_dim,))
    )
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU())

    model.add(keras.layers.Reshape((7, 7, 256)))

    model.add(
        keras.layers.Conv2DTranspose(
            128, kernel_size=(5, 5), strides=(1, 1), padding="same", use_bias=False
        )
    )
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU())

    model.add(
        keras.layers.Conv2DTranspose(
            64, kernel_size=(5, 5), strides=(2, 2), padding="same", use_bias=False
        )
    )
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU())

    model.add(
        keras.layers.Conv2DTranspose(
            1,
            kernel_size=(5, 5),
            strides=(2, 2),
            padding="same",
            use_bias=False,
            activation="tanh",
        )
    )
    model.build((None, noise_dim))
    return model


def generator_loss(fake_output: tf.Tensor) -> tf.Tensor:
    """Standard DCGAN generator loss using binary cross-entropy.

    Args:
        fake_output: Discriminator logits for generated (fake) images.

    Returns:
        Loss value encouraging the generator to fool the discriminator.
    """
    cross_entropy = keras.losses.BinaryCrossentropy(from_logits=True)
    return cross_entropy(tf.ones_like(fake_output), fake_output)
