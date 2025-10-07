"""Visualization helpers for the FedGAN prototype."""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_image_grid(
    images: np.ndarray,
    epoch: int,
    agent_id: int,
    output_dir: str,
    cmap: str = "gray",
) -> str:
    """Save a grid of generated images with an epoch label.

    Args:
        images: Array of shape (N, H, W[, C]) in the range [-1, 1] or [0, 1].
        epoch: Current training epoch.
        agent_id: Numeric agent identifier.
        output_dir: Directory path where the frame will be stored.
        cmap: Matplotlib colormap (default ``"gray"`` for MNIST).

    Returns:
        Absolute path to the saved PNG file.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    frame_path = output_path / f"image_epoch_{epoch:04d}.png"

    imgs = np.asarray(images)
    if imgs.ndim == 4 and imgs.shape[-1] == 1:
        imgs = imgs.squeeze(-1)

    # Rescale from [-1, 1] to [0, 1] if needed.
    if imgs.min() < 0:
        imgs = (imgs + 1.0) / 2.0
    imgs = np.clip(imgs, 0.0, 1.0)

    grid_cols = int(np.ceil(np.sqrt(len(imgs))))
    grid_rows = int(np.ceil(len(imgs) / grid_cols))

    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(grid_cols * 1.5, grid_rows * 1.5))
    axes = np.array(axes).reshape(grid_rows, grid_cols)

    for idx, ax in enumerate(axes.flat):
        ax.axis("off")
        if idx < len(imgs):
            ax.imshow(imgs[idx], cmap=cmap)

    fig.suptitle(f"Agent {agent_id} – Epoch {epoch}", fontsize=12)
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    fig.text(0.99, 0.01, f"epoch: {epoch}", ha="right", va="bottom", fontsize=10)
    fig.savefig(frame_path)
    plt.close(fig)
    return str(frame_path)


def make_gif(image_paths: Sequence[str], gif_path: str, duration: float = 0.5) -> str:
    """Create an animated GIF from saved image frames."""
    if not image_paths:
        raise ValueError("image_paths must not be empty")

    gif_file = Path(gif_path)
    gif_file.parent.mkdir(parents=True, exist_ok=True)

    frames: List[np.ndarray] = [imageio.v2.imread(path) for path in image_paths]
    imageio.mimsave(gif_file, frames, duration=duration)  # type: ignore[arg-type]
    return str(gif_file)


def export_loss_history(
    loss_history: Sequence[dict],
    output_dir: str,
    agent_id: int,
) -> tuple[str, str]:
    """Persist loss curves to CSV and PNG."""
    if not loss_history:
        raise ValueError("loss_history must not be empty")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(loss_history)
    csv_path = out_dir / f"agent_{agent_id}_loss.csv"
    png_path = out_dir / f"agent_{agent_id}_loss.png"

    df.to_csv(csv_path, index=False)

    plt.figure(figsize=(6, 4))
    plt.plot(df["epoch"], df["gen_loss"], label="generator", color="#1f77b4")
    plt.plot(df["epoch"], df["disc_loss"], label="discriminator", color="#ff7f0e")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Agent {agent_id} Losses")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(png_path)
    plt.close()

    return str(csv_path), str(png_path)
