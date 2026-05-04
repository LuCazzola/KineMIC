from torch import nn
import torch as th
from typing import List, Optional

class MLP(nn.Module):
    """
    A generalizable MLP with optional dropout.
    
    Args:
        layer_dims (List[int]): A list of output dimensions for each linear layer.
        dropout_prob (Optional[float]): The probability of an element to be zeroed.
                                        Applied after each hidden layer's activation.
                                        Default is None (no dropout).
    """
    def __init__(self, input_dim: int, layer_dims: List[int] = [256, 128], dropout_prob: float = 0.):
        super().__init__()
        
        layers = []
        current_dim = input_dim
        for i, dim in enumerate(layer_dims):
            layers.append(nn.Linear(current_dim, dim))
            if i < len(layer_dims) - 1:
                layers.append(nn.LeakyReLU(0.2, inplace=True))
                if dropout_prob > 0.:
                    layers.append(nn.Dropout(p=dropout_prob))
            
            current_dim = dim
            
        self.model = nn.Sequential(*layers)

    def forward(self, x: th.Tensor) -> th.Tensor:
        """Passes the input tensor through the MLP."""
        return self.model(x)