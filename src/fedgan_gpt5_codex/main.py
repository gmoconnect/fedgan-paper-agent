#!/usr/bin/env python3
"""CLI entry point for the FedGAN prototype implementation."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List

try:  # pragma: no cover - optional import path for script execution
    from .agents.agent import FedGANAgent
    from .server.intermediary import Intermediary
    from .utils.data import (
        FederatedDataset,
        build_federated_datasets,
        create_default_partitions,
    )
    from .utils.visualization import make_gif
except ImportError:  # pragma: no cover - fallback when run as a script
    CURRENT_DIR = Path(__file__).resolve().parent
    SRC_DIR = CURRENT_DIR.parent
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    from fedgan_gpt5_codex.agents.agent import FedGANAgent
    from fedgan_gpt5_codex.server.intermediary import Intermediary
    from fedgan_gpt5_codex.utils.data import (
        FederatedDataset,
        build_federated_datasets,
        create_default_partitions,
    )
    from fedgan_gpt5_codex.utils.visualization import make_gif

CURRENT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FedGAN prototype leveraging MNIST and DCGAN components"
    )
    parser.add_argument("--num-agents", type=int, default=3, help="Number of agents")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument(
        "--sync-interval",
        type=int,
        default=5,
        help="Local update steps K before synchronization",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for each agent dataset",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=10_000,
        help="Shuffle buffer size for agent datasets",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate shared by generator and discriminator optimizers",
    )
    parser.add_argument(
        "--beta1",
        type=float,
        default=0.5,
        help="Adam beta1 hyper-parameter matching DCGAN tutorial",
    )
    parser.add_argument(
        "--noise-dim",
        type=int,
        default=100,
        help="Latent vector dimensionality for the generator",
    )
    parser.add_argument(
        "--samples-to-generate",
        type=int,
        default=16,
        help="Number of samples generated per epoch for visualization",
    )
    parser.add_argument(
        "--visualization-seed",
        type=int,
        default=0,
        help="Base seed for deterministic visualization noise",
    )
    parser.add_argument(
        "--gif-duration",
        type=float,
        default=0.5,
        help="Frame duration (seconds) for the animated GIF",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CURRENT_DIR / "outputs",
        help="Directory used to store training artifacts",
    )
    parser.add_argument(
        "--classes-per-agent",
        type=str,
        default=None,
        help=(
            "JSON string such as '[[0,1,2],[3,4,5],[6,7,8,9]]' describing class"
            " allocation per agent. If omitted, a default partition is used."
        ),
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=None,
        help="Optional seed applied to dataset shuffling",
    )
    return parser.parse_args()


def parse_class_partition(arg: str | None, num_agents: int) -> List[List[int]]:
    if arg is None:
        return create_default_partitions(num_agents)
    try:
        partition = json.loads(arg)
    except json.JSONDecodeError as exc:  # pragma: no cover - CLI validation
        raise ValueError("Failed to parse --classes-per-agent JSON") from exc
    if not isinstance(partition, list) or len(partition) != num_agents:
        raise ValueError(
            "--classes-per-agent must describe a list with length equal to num-agents"
        )
    validated: List[List[int]] = []
    for idx, item in enumerate(partition):
        if not isinstance(item, list) or not item:
            raise ValueError(f"Partition for agent {idx} must be a non-empty list")
        if any(not isinstance(lbl, int) for lbl in item):
            raise ValueError(f"Partition for agent {idx} must contain integers only")
        validated.append(list(item))
    return validated


def initialize_agents(
    dataset: FederatedDataset,
    args: argparse.Namespace,
) -> List[FedGANAgent]:
    agents: List[FedGANAgent] = []
    for agent_id, (ds, size) in enumerate(zip(dataset.datasets, dataset.data_sizes)):
        agent = FedGANAgent(
            agent_id=agent_id,
            dataset=ds,
            data_size=size,
            noise_dim=args.noise_dim,
            learning_rate=args.learning_rate,
            beta_1=args.beta1,
            visualization_seed=args.visualization_seed,
            samples_to_generate=args.samples_to_generate,
        )
        agents.append(agent)
    return agents


def train(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)

    class_partition = parse_class_partition(args.classes_per_agent, args.num_agents)
    federated_dataset = build_federated_datasets(
        num_agents=args.num_agents,
        classes_per_agent=class_partition,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        shuffle_seed=args.shuffle_seed,
    )

    agents = initialize_agents(federated_dataset, args)
    intermediary = Intermediary(args.num_agents)

    image_paths: List[List[str]] = [[] for _ in range(args.num_agents)]

    print("=" * 80)
    print("FedGAN Prototype")
    print("=" * 80)
    print(f"Agents: {args.num_agents}")
    print(f"Epochs: {args.epochs}")
    print(f"Sync interval K: {args.sync_interval}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Output directory: {args.output_dir}")
    print("Class partitions:")
    for idx, classes in enumerate(class_partition):
        print(f"  Agent {idx}: {classes} ({federated_dataset.data_sizes[idx]} samples)")

    start = time.time()

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        epoch_losses: List[tuple[float, float]] = []

        for agent in agents:
            gen_loss, disc_loss = agent.train_local_steps(args.sync_interval)
            agent.record_epoch_loss(epoch, gen_loss, disc_loss)
            epoch_losses.append((gen_loss, disc_loss))
            print(
                f"  Agent {agent.agent_id}: gen_loss={gen_loss:.4f} disc_loss={disc_loss:.4f}"
            )

        agent_params = [agent.get_parameters() for agent in agents]
        aggregated_gen, aggregated_disc = intermediary.aggregate(
            agent_params, federated_dataset.data_sizes
        )

        for agent in agents:
            agent.set_parameters(aggregated_gen, aggregated_disc)

        print(f"  Synchronization round {intermediary.rounds_completed} complete")

        for agent in agents:
            agent_dir = args.output_dir / f"agent_{agent.agent_id}" / "images"
            img_path = agent.generate_visualization(epoch, str(agent_dir))
            image_paths[agent.agent_id].append(img_path)

    elapsed = time.time() - start
    print("\nTraining finished in {:.2f} seconds".format(elapsed))

    for agent in agents:
        agent_dir = args.output_dir / f"agent_{agent.agent_id}"
        csv_path, png_path = agent.save_losses(str(agent_dir))
        print(
            f"  Agent {agent.agent_id} loss history -> CSV: {csv_path}, plot: {png_path}"
        )

    for agent_id, paths in enumerate(image_paths):
        gif_path = args.output_dir / f"agent_{agent_id}" / "training_progress.gif"
        make_gif(paths, str(gif_path), duration=args.gif_duration)
        print(f"  Agent {agent_id} GIF saved to {gif_path}")

    print("=" * 80)
    print("Artifacts saved under:", args.output_dir)


if __name__ == "__main__":
    train(parse_args())
