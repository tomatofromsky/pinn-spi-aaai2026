"""
Normalization layers for neural networks.

Provides normalization/denormalization layers that map between raw domain
and normalized [-1, 1] domain using environment bounds (max_x, max_u).
"""

import torch
import torch.nn as nn


class StateNormalization(nn.Module):
    """
    Normalizes states from [-max_x, max_x] to [-1, 1].

    This layer should be used as the first layer in value/policy/Q networks
    to improve training stability by working in a normalized domain.

    Args:
        max_x: Maximum state bounds [d] or scalar
               States are assumed to be in [-max_x, max_x]
    """

    def __init__(self, max_x):
        super().__init__()
        # Convert to tensor if scalar
        if isinstance(max_x, (int, float)):
            max_x = torch.tensor([max_x])
        elif isinstance(max_x, torch.Tensor):
            max_x = max_x.clone()
        else:
            max_x = torch.tensor(max_x)

        # Register as buffer (not a learnable parameter, but part of state_dict)
        self.register_buffer("max_x", max_x.float())

    def forward(self, x):
        """
        Normalize state: x_normalized = x / max_x

        Args:
            x: State tensor [..., d]

        Returns:
            x_normalized: Normalized state in [-1, 1] domain
        """
        return x / self.max_x

    def inverse(self, x_normalized):
        """
        Denormalize state: x = x_normalized * max_x

        Args:
            x_normalized: Normalized state [..., d]

        Returns:
            x: State in original domain
        """
        return x_normalized * self.max_x


class ActionNormalization(nn.Module):
    """
    Normalizes actions from [-max_u, max_u] to [-1, 1] (forward)
    or denormalizes from [-1, 1] to [-max_u, max_u] (inverse).

    This layer should be used:
    - As first layer in Q-network action input (normalization)
    - As final layer in policy network output (denormalization)

    Args:
        max_u: Maximum action bounds [m] or scalar
               Actions are assumed to be in [-max_u, max_u]
    """

    def __init__(self, max_u):
        super().__init__()
        # Convert to tensor if scalar
        if isinstance(max_u, (int, float)):
            max_u = torch.tensor([max_u])
        elif isinstance(max_u, torch.Tensor):
            max_u = max_u.clone()
        else:
            max_u = torch.tensor(max_u)

        # Register as buffer (not a learnable parameter, but part of state_dict)
        self.register_buffer("max_u", max_u.float())

    def forward(self, u):
        """
        Normalize action: u_normalized = u / max_u

        Args:
            u: Action tensor [..., m]

        Returns:
            u_normalized: Normalized action in [-1, 1] domain
        """
        return u / self.max_u

    def inverse(self, u_normalized):
        """
        Denormalize action: u = u_normalized * max_u

        Args:
            u_normalized: Normalized action [..., m]

        Returns:
            u: Action in original domain
        """
        return u_normalized * self.max_u


class StateActionNormalization(nn.Module):
    """
    Combined normalization for state-action pairs (used in Q-networks).

    Normalizes states and actions independently before concatenation.

    Args:
        max_x: Maximum state bounds [d] or scalar
        max_u: Maximum action bounds [m] or scalar
    """

    def __init__(self, max_x, max_u):
        super().__init__()
        self.state_norm = StateNormalization(max_x)
        self.action_norm = ActionNormalization(max_u)

    def forward(self, x, u):
        """
        Normalize state-action pair and concatenate.

        Args:
            x: State tensor [..., d]
            u: Action tensor [..., m]

        Returns:
            xu_normalized: Normalized concatenated [x, u] tensor [..., d+m]
        """
        x_norm = self.state_norm(x)
        u_norm = self.action_norm(u)
        return torch.cat([x_norm, u_norm], dim=-1)
