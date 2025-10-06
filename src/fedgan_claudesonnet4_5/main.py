#!/usr/bin/env python3
"""FedGAN: Federated Generative Adversarial Network

Implementation of FedGAN algorithm based on the paper:
"Communication-Efficient Learning of Deep Networks from Decentralized Data"

This implementation follows Algorithm 1 from the paper, where:
- Each agent has local Generator and Discriminator
- Agents train locally for K steps
- Parameters are synchronized via intermediary server using weighted averaging
- Data is partitioned across agents (non-IID distribution)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.agent import Agent
from server.intermediary import Intermediary
from utils.data_loader import load_mnist_partitioned, create_default_partition
from utils.visualization import make_gif


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='FedGAN: Federated Generative Adversarial Network'
    )
    
    # FedGAN specific parameters
    parser.add_argument(
        '--num_agents',
        type=int,
        default=3,
        help='Number of agents (default: 3)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Total number of training epochs (default: 50)'
    )
    parser.add_argument(
        '--sync_interval',
        type=int,
        default=5,
        help='Synchronization interval K (default: 5)'
    )
    
    # GAN parameters
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for training (default: 32)'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-4,
        help='Learning rate for optimizers (default: 1e-4)'
    )
    parser.add_argument(
        '--noise_dim',
        type=int,
        default=100,
        help='Dimension of generator noise vector (default: 100)'
    )
    parser.add_argument(
        '--num_examples_to_generate',
        type=int,
        default=16,
        help='Number of images to generate per epoch (default: 16)'
    )
    
    # Output parameters
    parser.add_argument(
        '--output_dir',
        default='tmp_outputs',
        help='Directory to save outputs (default: tmp_outputs)'
    )
    parser.add_argument(
        '--gif_duration',
        type=float,
        default=0.5,
        help='Frame duration in seconds for GIF (default: 0.5)'
    )
    parser.add_argument(
        '--visualization_seed',
        type=int,
        default=42,
        help='Base seed for visualization (default: 42)'
    )
    
    return parser.parse_args()


def main() -> None:
    """Main training loop for FedGAN."""
    args = parse_args()
    
    print("=" * 80)
    print("FedGAN: Federated Generative Adversarial Network")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Number of agents: {args.num_agents}")
    print(f"  Total epochs: {args.epochs}")
    print(f"  Synchronization interval K: {args.sync_interval}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Output directory: {args.output_dir}")
    print("=" * 80)
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Partition MNIST data across agents
    print("\nLoading and partitioning MNIST dataset...")
    classes_per_agent = create_default_partition(args.num_agents)
    datasets, data_sizes = load_mnist_partitioned(
        num_agents=args.num_agents,
        classes_per_agent=classes_per_agent,
        batch_size=args.batch_size
    )
    
    # Create agents
    print("\nInitializing agents...")
    agents: List[Agent] = []
    for agent_id in range(args.num_agents):
        agent = Agent(
            agent_id=agent_id,
            local_data=datasets[agent_id],
            data_size=data_sizes[agent_id],
            noise_dim=args.noise_dim,
            learning_rate=args.learning_rate,
            num_examples_to_generate=args.num_examples_to_generate,
            visualization_seed=args.visualization_seed
        )
        agents.append(agent)
        print(f"  Agent {agent_id} initialized with {data_sizes[agent_id]} samples")
    
    # Create intermediary server
    intermediary = Intermediary(num_agents=args.num_agents)
    print(f"\nIntermediary server initialized for {args.num_agents} agents")
    
    # Training loop
    print("\n" + "=" * 80)
    print("Starting FedGAN training...")
    print("=" * 80)
    
    start_time = time.time()
    
    # Track image paths for GIF creation
    agent_image_paths: List[List[str]] = [[] for _ in range(args.num_agents)]
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 80)
        
        # Local training phase (K steps)
        # Each agent trains on their local data
        epoch_losses = []
        for agent in agents:
            avg_gen_loss, avg_disc_loss = agent.train_local_epochs(
                num_steps=args.sync_interval
            )
            agent.record_loss(epoch, avg_gen_loss, avg_disc_loss)
            epoch_losses.append((avg_gen_loss, avg_disc_loss))
            print(f"  Agent {agent.agent_id}: "
                  f"gen_loss={avg_gen_loss:.4f}, disc_loss={avg_disc_loss:.4f}")
        
        # Synchronization phase (every K steps)
        # Agents send parameters to intermediary
        agent_params = [agent.get_parameters() for agent in agents]
        
        # Intermediary aggregates parameters using weighted averaging
        aggregated_gen_weights, aggregated_disc_weights = \
            intermediary.aggregate_parameters(agent_params, data_sizes)
        
        # Intermediary distributes aggregated parameters to all agents
        for agent in agents:
            agent.set_parameters(aggregated_gen_weights, aggregated_disc_weights)
        
        print(f"  ✓ Parameters synchronized (round {intermediary.get_sync_rounds()})")
        
        # Generate images for visualization (using synchronized parameters)
        for agent in agents:
            agent_output_dir = os.path.join(
                args.output_dir, f'agent_{agent.agent_id}', 'images'
            )
            img_path = agent.generate_images(epoch, agent_output_dir)
            agent_image_paths[agent.agent_id].append(img_path)
    
    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"Training complete in {total_time:.2f}s")
    print("=" * 80)
    
    # Save loss histories
    print("\nSaving loss histories...")
    for agent in agents:
        agent_output_dir = os.path.join(args.output_dir, f'agent_{agent.agent_id}')
        agent.save_losses(agent_output_dir)
        print(f"  Agent {agent.agent_id} losses saved to {agent_output_dir}")
    
    # Create GIFs for each agent
    print("\nCreating training progress GIFs...")
    for agent_id, image_paths in enumerate(agent_image_paths):
        gif_path = os.path.join(
            args.output_dir, f'agent_{agent_id}', 'training_progress.gif'
        )
        make_gif(image_paths, gif_path, duration=args.gif_duration)
        print(f"  Agent {agent_id} GIF: {gif_path}")
    
    print("\n" + "=" * 80)
    print("FedGAN training finished successfully!")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
