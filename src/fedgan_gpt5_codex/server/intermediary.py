"""FedGAN intermediary (server) that aggregates agent parameters."""
from __future__ import annotations

from typing import Sequence, Tuple
import numpy as np


class Intermediary:
    """Aggregate generator/discriminator parameters via weighted averaging."""

    def __init__(self, num_agents: int):
        if num_agents <= 0:
            raise ValueError("num_agents must be positive")
        self.num_agents = num_agents
        self._sync_rounds = 0

    def aggregate(
        self,
        agent_params: Sequence[Tuple[Sequence[np.ndarray], Sequence[np.ndarray]]],
        data_sizes: Sequence[int],
    ) -> Tuple[list[np.ndarray], list[np.ndarray]]:
        if len(agent_params) != self.num_agents:
            raise ValueError(
                f"Expected {self.num_agents} parameter sets, received {len(agent_params)}"
            )
        if len(data_sizes) != self.num_agents:
            raise ValueError(
                f"Expected {self.num_agents} data sizes, received {len(data_sizes)}"
            )
        total = float(sum(data_sizes))
        if total <= 0:
            raise ValueError("Total data size must be positive for aggregation")
        weights = [size / total for size in data_sizes]

        gen_weight_groups = [params[0] for params in agent_params]
        disc_weight_groups = [params[1] for params in agent_params]

        averaged_gen = self._weighted_average(gen_weight_groups, weights)
        averaged_disc = self._weighted_average(disc_weight_groups, weights)
        self._sync_rounds += 1
        return averaged_gen, averaged_disc

    def _weighted_average(
        self, weight_groups: Sequence[Sequence[np.ndarray]], weights: Sequence[float]
    ) -> list[np.ndarray]:
        num_layers = len(weight_groups[0])
        averaged: list[np.ndarray] = []
        for layer_idx in range(num_layers):
            stacked = np.stack(
                [np.asarray(agent_weights[layer_idx]) for agent_weights in weight_groups]
            )
            combined = np.zeros_like(stacked[0])
            for agent_idx, agent_weight in enumerate(weights):
                combined += agent_weight * stacked[agent_idx]
            averaged.append(combined)
        return averaged

    @property
    def rounds_completed(self) -> int:
        return self._sync_rounds
