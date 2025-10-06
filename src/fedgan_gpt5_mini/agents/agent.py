import tensorflow as tf
from ..models.generator import make_generator_model
from ..models.discriminator import make_discriminator_model

class Agent:
    def __init__(self, agent_id, data_ds, data_size, noise_dim=100, gen_lr=2e-4, disc_lr=2e-4):
        self.id = agent_id
        self.dataset = data_ds
        self.data_size = data_size
        self.noise_dim = noise_dim

        self.generator = make_generator_model(noise_dim)
        self.discriminator = make_discriminator_model()

        self.gen_optimizer = tf.keras.optimizers.Adam(gen_lr, beta_1=0.5)
        self.disc_optimizer = tf.keras.optimizers.Adam(disc_lr, beta_1=0.5)

        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        # tracking
        self.gen_losses = []
        self.disc_losses = []

    def set_weights(self, gen_weights, disc_weights):
        self.generator.set_weights(gen_weights)
        self.discriminator.set_weights(disc_weights)

    def get_weights(self):
        return self.generator.get_weights(), self.discriminator.get_weights()

    def train_one_epoch(self, seed, batch_random_seed=True):
        # Single epoch over local dataset
        # If batch_random_seed True, generator noise is random per batch; seed used for evaluation only
        gen_losses = []
        disc_losses = []
        for image_batch in self.dataset:
            gen_loss, disc_loss = self._train_step(image_batch)
            gen_losses.append(float(gen_loss))
            disc_losses.append(float(disc_loss))

        # record per-epoch averages
        if len(gen_losses) > 0:
            self.gen_losses.append(sum(gen_losses) / len(gen_losses))
            self.disc_losses.append(sum(disc_losses) / len(disc_losses))
        else:
            self.gen_losses.append(None)
            self.disc_losses.append(None)

        return self.gen_losses[-1], self.disc_losses[-1]

    def _train_step(self, images):
        noise = tf.random.normal([tf.shape(images)[0], self.noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)

            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            gen_loss = self._generator_loss(fake_output)
            disc_loss = self._discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.gen_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.disc_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        return gen_loss, disc_loss

    def _generator_loss(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)

    def _discriminator_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        return real_loss + fake_loss
