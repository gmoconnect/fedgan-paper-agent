"""Dataset utilities for the FedGAN prototype."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence
import numpy as np
import tensorflow as tf
from tensorflow import keras  # type: ignore[attr-defined]


@dataclass(frozen=True)
class FederatedDataset:
    """Container holding per-agent dataset information."""

    datasets: Sequence[tf.data.Dataset]
    data_sizes: Sequence[int]
    class_partitions: Sequence[Sequence[int]]


def create_default_partitions(num_agents: int) -> List[List[int]]:
    """Distribute MNIST digit classes as evenly as possible among agents."""
    if num_agents <= 0:
        raise ValueError("num_agents must be positive")

    classes = list(range(10))
    base, remainder = divmod(len(classes), num_agents)
    partitions: List[List[int]] = []
    start = 0
    for agent_idx in range(num_agents):
        count = base + (1 if agent_idx < remainder else 0)
        end = start + count
        partitions.append(classes[start:end])
        start = end
    # Guard against missing classes when num_agents > 10
    while len(partitions) < num_agents:
        partitions.append(classes.copy())
    return partitions


def build_federated_datasets(
    num_agents: int,
    classes_per_agent: Sequence[Sequence[int]],
    batch_size: int = 32,
    buffer_size: int = 10_000,
    shuffle_seed: int | None = None,
) -> FederatedDataset:
    """Create TensorFlow datasets per agent using MNIST digits."""
    if len(classes_per_agent) != num_agents:
        raise ValueError(
            "classes_per_agent must provide assignments for each agent"
        )

    (train_images, train_labels), _ = keras.datasets.mnist.load_data()

    train_images = train_images.reshape((-1, 28, 28, 1)).astype("float32")
    train_images = (train_images - 127.5) / 127.5

    datasets: List[tf.data.Dataset] = []
    sizes: List[int] = []

    for agent_idx, classes in enumerate(classes_per_agent):
        if not classes:
            raise ValueError(f"Agent {agent_idx} partition is empty")
        mask = np.isin(train_labels, classes)
        images = train_images[mask]
        sizes.append(int(images.shape[0]))

        ds = tf.data.Dataset.from_tensor_slices(images)
        ds = ds.shuffle(buffer_size, seed=shuffle_seed, reshuffle_each_iteration=True)
        ds = ds.batch(batch_size, drop_remainder=False)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        ds = ds.repeat()
        datasets.append(ds)

    return FederatedDataset(
        datasets=datasets,
        data_sizes=sizes,
        class_partitions=[tuple(partition) for partition in classes_per_agent],
    )
