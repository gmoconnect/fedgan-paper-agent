"""Visualization utilities for FedGAN

Functions for creating GIFs and visualizing training progress.
"""
import os
from typing import List
import imageio


def make_gif(
    image_paths: List[str],
    gif_path: str,
    duration: float = 0.5
) -> None:
    """Create animated GIF from a list of image paths.
    
    Args:
        image_paths: List of paths to PNG images
        gif_path: Output path for GIF file
        duration: Duration per frame in seconds
    """
    images = []
    for p in image_paths:
        images.append(imageio.v2.imread(p))
    
    os.makedirs(os.path.dirname(gif_path), exist_ok=True)
    imageio.mimsave(gif_path, images, duration=duration)
    print(f"GIF saved to {gif_path}")
