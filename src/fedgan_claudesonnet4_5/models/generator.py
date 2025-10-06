"""Generator model for FedGAN

Based on DCGAN tutorial architecture.
Generates 28x28x1 images from 100-dimensional noise vectors.
"""
import tensorflow as tf


def make_generator_model() -> tf.keras.Sequential:
    """Create generator model following DCGAN tutorial architecture.
    
    Returns:
        tf.keras.Sequential: Generator model that takes noise vector (100,) 
                            and outputs image (28, 28, 1)
    """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(100,)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size

    model.add(
        tf.keras.layers.Conv2DTranspose(
            128, (5, 5), strides=(1, 1), padding='same', use_bias=False
        )
    )
    assert model.output_shape == (None, 7, 7, 128)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(
        tf.keras.layers.Conv2DTranspose(
            64, (5, 5), strides=(2, 2), padding='same', use_bias=False
        )
    )
    assert model.output_shape == (None, 14, 14, 64)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(
        tf.keras.layers.Conv2DTranspose(
            1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'
        )
    )
    assert model.output_shape == (None, 28, 28, 1)

    return model


def generator_loss(fake_output: tf.Tensor) -> tf.Tensor:
    """Calculate generator loss.
    
    Generator wants discriminator to classify fake images as real (ones).
    
    Args:
        fake_output: Discriminator output for fake images
        
    Returns:
        Generator loss value
    """
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    return cross_entropy(tf.ones_like(fake_output), fake_output)
