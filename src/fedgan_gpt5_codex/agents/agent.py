"""FedGAN agent implementation for the prototype."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, List, Sequence, Tuple, cast

import numpy as np
import tensorflow as tf
from tensorflow import keras  # type: ignore[attr-defined]

from ..models.generator import build_generator, generator_loss
from ..models.discriminator import build_discriminator, discriminator_loss
from ..utils.visualization import export_loss_history, save_image_grid


@dataclass
class LossRecord:
    epoch: int
    gen_loss: float
    disc_loss: float


@dataclass
class FedGANAgent:
    """Single FedGAN agent with local models and data iterator."""

    agent_id: int
    dataset: tf.data.Dataset
    data_size: int
    noise_dim: int
    learning_rate: float
    beta_1: float
    visualization_seed: int
    samples_to_generate: int
    loss_history: List[LossRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.generator = build_generator(self.noise_dim)
        self.discriminator = build_discriminator()

        self.generator_optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate, beta_1=self.beta_1
        )
        self.discriminator_optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate, beta_1=self.beta_1
        )

        rng = np.random.default_rng(self.visualization_seed + self.agent_id)
        noise = rng.normal(size=(self.samples_to_generate, self.noise_dim)).astype(
            np.float32
        )
        self.fixed_noise = tf.convert_to_tensor(noise)
        self._dataset_iterator: Iterator[tf.Tensor] = iter(self.dataset)

    def _train_step(self, images: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        noise = tf.random.normal([tf.shape(images)[0], self.noise_dim])  # type: ignore[index]

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)

            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(
            gen_loss, self.generator.trainable_variables
        )
        gradients_of_discriminator = disc_tape.gradient(
            disc_loss, self.discriminator.trainable_variables
        )

        gen_grads = cast(Sequence[tf.Tensor | None], gradients_of_generator)
        disc_grads = cast(Sequence[tf.Tensor | None], gradients_of_discriminator)

        gen_pairs = [
            (grad, var)
            for grad, var in zip(gen_grads, self.generator.trainable_variables)
            if grad is not None
        ]
        disc_pairs = [
            (grad, var)
            for grad, var in zip(
                disc_grads, self.discriminator.trainable_variables
            )
            if grad is not None
        ]

        if gen_pairs:
            self.generator_optimizer.apply_gradients(gen_pairs)
        if disc_pairs:
            self.discriminator_optimizer.apply_gradients(disc_pairs)

        return gen_loss, disc_loss

    def train_local_steps(self, steps: int) -> Tuple[float, float]:
        """Run local GAN updates for the requested number of batches."""
        gen_losses: List[float] = []
        disc_losses: List[float] = []

        for _ in range(steps):
            try:
                batch = next(self._dataset_iterator)
            except StopIteration:
                self._dataset_iterator = iter(self.dataset)
                batch = next(self._dataset_iterator)

            batch_tensor = cast(tf.Tensor, batch)
            gen_loss, disc_loss = self._train_step(batch_tensor)
            gen_loss = cast(tf.Tensor, gen_loss)
            disc_loss = cast(tf.Tensor, disc_loss)
            gen_losses.append(float(gen_loss))  # type: ignore[arg-type]
            disc_losses.append(float(disc_loss))  # type: ignore[arg-type]

        return float(np.mean(gen_losses)), float(np.mean(disc_losses))

    def get_parameters(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        gen_weights = [np.array(w) for w in self.generator.get_weights()]
        disc_weights = [np.array(w) for w in self.discriminator.get_weights()]
        return gen_weights, disc_weights

    def set_parameters(
        self, gen_weights: Sequence[np.ndarray], disc_weights: Sequence[np.ndarray]
    ) -> None:
        self.generator.set_weights(gen_weights)
        self.discriminator.set_weights(disc_weights)

    def generate_visualization(self, epoch: int, output_dir: str) -> str:
        predictions = self.generator(self.fixed_noise, training=False)
        images = predictions.numpy()
        return save_image_grid(images, epoch, self.agent_id, output_dir)

    def record_epoch_loss(self, epoch: int, gen_loss: float, disc_loss: float) -> None:
        self.loss_history.append(LossRecord(epoch, gen_loss, disc_loss))

    def save_losses(self, output_dir: str) -> Tuple[str, str]:
        history_dicts = [
            {"epoch": record.epoch, "gen_loss": record.gen_loss, "disc_loss": record.disc_loss}
            for record in self.loss_history
        ]
        return export_loss_history(history_dicts, output_dir, self.agent_id)
