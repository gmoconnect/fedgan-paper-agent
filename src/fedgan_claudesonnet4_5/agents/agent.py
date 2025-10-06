"""Agent class for FedGAN

Each agent has local Generator and Discriminator, trains on local data,
and can send/receive parameters to/from intermediary server.
"""
from typing import Tuple, List
import os
import numpy as np
import tensorflow as tf

from models.generator import make_generator_model, generator_loss
from models.discriminator import make_discriminator_model, discriminator_loss


class Agent:
    """FedGAN Agent with local Generator and Discriminator.
    
    Each agent trains on its local dataset partition and periodically
    synchronizes parameters with the intermediary server.
    """
    
    def __init__(
        self,
        agent_id: int,
        local_data: tf.data.Dataset,
        data_size: int,
        noise_dim: int = 100,
        learning_rate: float = 1e-4,
        num_examples_to_generate: int = 16,
        visualization_seed: int = 42
    ):
        """Initialize agent with local models and data.
        
        Args:
            agent_id: Unique identifier for this agent
            local_data: TensorFlow dataset containing local training data
            data_size: Number of samples in local dataset
            noise_dim: Dimension of noise vector for generator
            learning_rate: Learning rate for optimizers
            num_examples_to_generate: Number of images to generate per epoch
            visualization_seed: Seed for fixed visualization noise (for GIF generation)
        """
        self.agent_id = agent_id
        self.local_data = local_data
        self.data_size = data_size
        self.noise_dim = noise_dim
        
        # Create local models
        self.generator = make_generator_model()
        self.discriminator = make_discriminator_model()
        
        # Create optimizers
        self.generator_optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate, beta_1=0.5
        )
        self.discriminator_optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate, beta_1=0.5
        )
        
        # Fixed seed for visualization (to track progress across epochs)
        np.random.seed(visualization_seed + agent_id)  # Different seed per agent
        fixed_seed = np.random.normal(
            size=(num_examples_to_generate, noise_dim)
        ).astype(np.float32)
        self.fixed_seed = tf.convert_to_tensor(fixed_seed)
        
        # Loss history
        self.loss_history: List[dict] = []
        
    @tf.function
    def train_step(self, images: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Perform one training step on a batch of images.
        
        Args:
            images: Batch of real images
            
        Returns:
            Tuple of (generator_loss, discriminator_loss)
        """
        # Random noise for this batch
        noise = tf.random.normal([tf.shape(images)[0], self.noise_dim])
        
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
        
        self.generator_optimizer.apply_gradients(
            zip(gradients_of_generator, self.generator.trainable_variables)
        )
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables)
        )
        
        return gen_loss, disc_loss
    
    def train_local_epochs(self, num_steps: int) -> Tuple[float, float]:
        """Train locally for specified number of steps (K in FedGAN algorithm).
        
        Args:
            num_steps: Number of local update steps (K)
            
        Returns:
            Tuple of (average_gen_loss, average_disc_loss) over all steps
        """
        gen_losses = []
        disc_losses = []
        
        step_count = 0
        for image_batch in self.local_data:
            if step_count >= num_steps:
                break
                
            g_loss, d_loss = self.train_step(image_batch)
            gen_losses.append(float(g_loss.numpy()))
            disc_losses.append(float(d_loss.numpy()))
            step_count += 1
        
        # Compute averages
        avg_gen_loss = sum(gen_losses) / len(gen_losses) if gen_losses else 0.0
        avg_disc_loss = sum(disc_losses) / len(disc_losses) if disc_losses else 0.0
        
        return avg_gen_loss, avg_disc_loss
    
    def get_parameters(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Get current model parameters for synchronization.
        
        Returns:
            Tuple of (generator_weights, discriminator_weights)
        """
        gen_weights = [w.numpy() for w in self.generator.get_weights()]
        disc_weights = [w.numpy() for w in self.discriminator.get_weights()]
        return gen_weights, disc_weights
    
    def set_parameters(
        self, 
        gen_weights: List[np.ndarray], 
        disc_weights: List[np.ndarray]
    ) -> None:
        """Set model parameters from intermediary server.
        
        Args:
            gen_weights: Generator weights to set
            disc_weights: Discriminator weights to set
        """
        self.generator.set_weights(gen_weights)
        self.discriminator.set_weights(disc_weights)
    
    def generate_images(self, epoch: int, output_dir: str) -> str:
        """Generate images using fixed seed for visualization.
        
        Args:
            epoch: Current epoch number
            output_dir: Directory to save generated images
            
        Returns:
            Path to saved image file
        """
        import matplotlib.pyplot as plt
        
        # Generate images
        predictions = self.generator(self.fixed_seed, training=False)
        
        # Rescale from [-1,1] to [0,255]
        imgs = (predictions.numpy() * 127.5 + 127.5).astype(np.uint8)
        imgs = imgs.reshape((-1, 28, 28))
        
        n = int(np.ceil(np.sqrt(imgs.shape[0])))
        fig = plt.figure(figsize=(n, n))
        
        for i in range(imgs.shape[0]):
            plt.subplot(n, n, i + 1)
            plt.imshow(imgs[i], cmap='gray')
            plt.axis('off')
        plt.suptitle(f'Agent {self.agent_id} - Epoch {epoch}', fontsize=12)
        
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f'image_at_epoch_{epoch:04d}.png')
        plt.savefig(path)
        plt.close(fig)
        return path
    
    def record_loss(self, epoch: int, gen_loss: float, disc_loss: float) -> None:
        """Record loss values for this epoch.
        
        Args:
            epoch: Current epoch number
            gen_loss: Generator loss
            disc_loss: Discriminator loss
        """
        self.loss_history.append({
            'epoch': epoch,
            'gen_loss': gen_loss,
            'disc_loss': disc_loss
        })
    
    def save_losses(self, output_dir: str) -> None:
        """Save loss history to CSV and PNG.
        
        Args:
            output_dir: Directory to save loss files
        """
        import pandas as pd
        import matplotlib.pyplot as plt
        
        loss_df = pd.DataFrame(self.loss_history)
        
        # Save CSV
        csv_path = os.path.join(output_dir, f'agent_{self.agent_id}_losses.csv')
        loss_df.to_csv(csv_path, index=False)
        
        # Save plot
        plt.figure()
        plt.plot(loss_df['epoch'], loss_df['gen_loss'], label='gen_loss')
        plt.plot(loss_df['epoch'], loss_df['disc_loss'], label='disc_loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title(f'Agent {self.agent_id} Losses')
        plt.legend()
        plt.grid(True)
        
        png_path = os.path.join(output_dir, f'agent_{self.agent_id}_losses.png')
        plt.savefig(png_path)
        plt.close()
