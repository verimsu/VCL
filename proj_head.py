import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from typing import List, Optional, Sequence, Tuple

class VCLProjectionHead(nn.Module):
    """Projection head used for SimCLR with two outputs: mu and logVar."""

    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 2048,
        output_dim: int = 128,
        num_layers: int = 3,
        batch_norm: bool = True,
    ):
        """Initialize a new SimCLRProjectionHead instance with two outputs.

        Args:
            input_dim: Number of input dimensions.
            hidden_dim: Number of hidden dimensions.
            output_dim: Number of output dimensions for mu and logVar.
            num_layers: Number of hidden layers (2 for v1, 3+ for v2).
            batch_norm: Whether or not to use batch norms.
        """
        super().__init__()

        layers: List[nn.Module] = []
        # Add initial layer
        layers.append(
            nn.Linear(input_dim, hidden_dim, bias=not batch_norm)
        )
        if batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())

        # Add additional hidden layers if needed
        for _ in range(2, num_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=not batch_norm))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())

        # Sequential container for hidden layers
        self.hidden_layers = nn.Sequential(*layers)

        # Separate linear layers for mu and logVar
        self.mu_layer = nn.Linear(hidden_dim, output_dim)
        self.logvar_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes forward pass and returns mu and logVar.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            A tuple of tensors (mu, logVar) each of shape (batch_size, output_dim).
        """
        # Pass through hidden layers
        hidden_representation = self.hidden_layers(x)

        # Compute mu and logVar
        mu = self.mu_layer(hidden_representation)
        logVar = self.logvar_layer(hidden_representation)
        
        dist = Normal(mu, (logVar / 2).exp())
        z = dist.rsample()

        return z, mu, logVar
