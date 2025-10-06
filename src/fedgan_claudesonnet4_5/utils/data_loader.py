"""Data utilities for FedGAN

Functions for loading and partitioning MNIST dataset across agents.
"""
from typing import List, Tuple
import numpy as np
import tensorflow as tf


def load_mnist_partitioned(
    num_agents: int,
    classes_per_agent: List[List[int]],
    batch_size: int = 32,
    buffer_size: int = 10000
) -> Tuple[List[tf.data.Dataset], List[int]]:
    """Load MNIST and partition across agents by class labels.
    
    Implements non-IID data distribution by assigning different digit
    classes to different agents.
    
    Args:
        num_agents: Number of agents to partition data for
        classes_per_agent: List of class lists for each agent.
                          e.g., [[0,1,2], [3,4,5], [6,7,8,9]]
        batch_size: Batch size for training
        buffer_size: Shuffle buffer size per agent
        
    Returns:
        Tuple of (datasets, data_sizes) where:
        - datasets: List of TensorFlow datasets, one per agent
        - data_sizes: List of dataset sizes for each agent
    """
    if len(classes_per_agent) != num_agents:
        raise ValueError(
            f"classes_per_agent length ({len(classes_per_agent)}) must match "
            f"num_agents ({num_agents})"
        )
    
    # Load full MNIST dataset
    (train_images, train_labels), _ = tf.keras.datasets.mnist.load_data()
    
    # Normalize to [-1, 1] to match generator output (tanh activation)
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    train_images = (train_images - 127.5) / 127.5
    
    datasets = []
    data_sizes = []
    
    for agent_idx in range(num_agents):
        # Get classes assigned to this agent
        agent_classes = classes_per_agent[agent_idx]
        
        # Filter images for this agent's classes
        mask = np.isin(train_labels, agent_classes)
        agent_images = train_images[mask]
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices(agent_images)
        dataset = dataset.shuffle(buffer_size).batch(batch_size)
        
        datasets.append(dataset)
        data_sizes.append(len(agent_images))
        
        print(f"Agent {agent_idx}: classes={agent_classes}, "
              f"samples={len(agent_images)}")
    
    return datasets, data_sizes


def create_default_partition(num_agents: int) -> List[List[int]]:
    """Create default class partition for MNIST (10 classes).
    
    Distributes 10 digit classes as evenly as possible across agents.
    
    Args:
        num_agents: Number of agents
        
    Returns:
        List of class assignments for each agent
    """
    all_classes = list(range(10))
    classes_per_agent = []
    
    # Distribute classes evenly
    classes_per_split = len(all_classes) // num_agents
    remainder = len(all_classes) % num_agents
    
    start_idx = 0
    for i in range(num_agents):
        # Give extra class to first 'remainder' agents
        num_classes = classes_per_split + (1 if i < remainder else 0)
        end_idx = start_idx + num_classes
        classes_per_agent.append(all_classes[start_idx:end_idx])
        start_idx = end_idx
    
    return classes_per_agent
