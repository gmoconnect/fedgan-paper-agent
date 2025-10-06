"""Intermediary (Server) for FedGAN

Aggregates parameters from multiple agents using weighted averaging
based on local data sizes.
"""
from typing import List, Tuple
import numpy as np


class Intermediary:
    """FedGAN Intermediary server that aggregates agent parameters.
    
    The server collects parameters from all agents, computes a weighted
    average based on each agent's data size, and distributes the averaged
    parameters back to all agents.
    """
    
    def __init__(self, num_agents: int):
        """Initialize intermediary.
        
        Args:
            num_agents: Number of agents participating in federated training
        """
        self.num_agents = num_agents
        self.total_sync_rounds = 0
        
    def aggregate_parameters(
        self,
        agent_params: List[Tuple[List[np.ndarray], List[np.ndarray]]],
        data_sizes: List[int]
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Aggregate parameters from all agents using weighted averaging.
        
        Implements the synchronization phase of FedGAN Algorithm 1:
        - w_n = Σ(j=1 to B) p_j * w_n^j
        - θ_n = Σ(j=1 to B) p_j * θ_n^j
        where p_j = |R_j| / Σ|R_k| (data size ratio)
        
        Args:
            agent_params: List of (generator_weights, discriminator_weights) 
                         from each agent
            data_sizes: List of data sizes for each agent
            
        Returns:
            Tuple of (averaged_gen_weights, averaged_disc_weights)
        """
        if len(agent_params) != self.num_agents:
            raise ValueError(
                f"Expected {self.num_agents} agents, got {len(agent_params)}"
            )
        
        if len(data_sizes) != self.num_agents:
            raise ValueError(
                f"Expected {self.num_agents} data sizes, got {len(data_sizes)}"
            )
        
        # Calculate weights based on data size
        total_data = sum(data_sizes)
        weights = [size / total_data for size in data_sizes]
        
        # Extract generator and discriminator weights
        gen_weights_list = [params[0] for params in agent_params]
        disc_weights_list = [params[1] for params in agent_params]
        
        # Aggregate generator weights
        aggregated_gen_weights = self._weighted_average(
            gen_weights_list, weights
        )
        
        # Aggregate discriminator weights
        aggregated_disc_weights = self._weighted_average(
            disc_weights_list, weights
        )
        
        self.total_sync_rounds += 1
        
        return aggregated_gen_weights, aggregated_disc_weights
    
    def _weighted_average(
        self,
        weights_list: List[List[np.ndarray]],
        weights: List[float]
    ) -> List[np.ndarray]:
        """Compute weighted average of model weights.
        
        Args:
            weights_list: List of weight lists from each agent
            weights: Weight for each agent (based on data size)
            
        Returns:
            List of averaged weight arrays
        """
        num_layers = len(weights_list[0])
        averaged_weights = []
        
        for layer_idx in range(num_layers):
            # Stack weights from all agents for this layer
            layer_weights = np.array([
                agent_weights[layer_idx] for agent_weights in weights_list
            ])
            
            # Compute weighted average
            # weights shape: (num_agents,)
            # layer_weights shape: (num_agents, *weight_shape)
            weighted_sum = np.zeros_like(layer_weights[0])
            for agent_idx, w in enumerate(weights):
                weighted_sum += w * layer_weights[agent_idx]
            
            averaged_weights.append(weighted_sum)
        
        return averaged_weights
    
    def get_sync_rounds(self) -> int:
        """Get total number of synchronization rounds completed.
        
        Returns:
            Number of sync rounds
        """
        return self.total_sync_rounds
