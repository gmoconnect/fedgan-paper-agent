"""Discriminator model for FedGAN

Based on DCGAN tutorial architecture.
Classifies 28x28x1 images as real or fake.
"""
import tensorflow as tf


def make_discriminator_model() -> tf.keras.Sequential:
    """Create discriminator model following DCGAN tutorial architecture.
    
    Returns:
        tf.keras.Sequential: Discriminator model that takes image (28, 28, 1)
                            and outputs classification logit
    """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', 
                                      input_shape=[28, 28, 1]))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1))

    return model


def discriminator_loss(real_output: tf.Tensor, fake_output: tf.Tensor) -> tf.Tensor:
    """Calculate discriminator loss.
    
    Discriminator wants to classify real images as real (ones) and 
    fake images as fake (zeros).
    
    Args:
        real_output: Discriminator output for real images
        fake_output: Discriminator output for fake images
        
    Returns:
        Discriminator loss value
    """
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss
