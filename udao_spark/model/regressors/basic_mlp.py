from dataclasses import dataclass
from typing import List

import torch as th
import torch.nn as nn
from udao.model import BaseRegressor


class BasicMLP(BaseRegressor):
    """
    Basic MLP regressor, mimicing AutoGluon's EmbedNet.
    """

    @dataclass
    class Params(BaseRegressor.Params):
        hidden_dim: int
        """Size of the hidden layers outputs."""
        n_layers: int
        """Number of layers in the MLP"""
        dropout: float
        """Probability of dropout."""
        use_batchnorm: bool = True
        """Whether to use batch normalization."""
        activation: str = "relu"
        """Activation function."""

    def _load_layers(self, net_params: Params) -> nn.ModuleList:
        """Create the list of fully connected layers."""

        act_fn: nn.Module
        if net_params.activation == "relu":
            act_fn = nn.ReLU()
        elif net_params.activation == "tanh":
            act_fn = nn.Tanh()
        elif net_params.activation == "elu":
            act_fn = nn.ELU()
        else:
            raise ValueError(f"Unknown activation function: {net_params.activation}")

        layers: List[nn.Module] = []
        if net_params.use_batchnorm:
            layers.append(nn.BatchNorm1d(self.input_dim))
        layers.append(nn.Linear(self.input_dim, net_params.hidden_dim))
        layers.append(act_fn)
        for _ in range(net_params.n_layers - 1):
            if net_params.use_batchnorm:
                layers.append(nn.BatchNorm1d(net_params.hidden_dim))
            layers.append(nn.Linear(net_params.hidden_dim, net_params.hidden_dim))
            layers.append(act_fn)
        layers.append(nn.Linear(net_params.hidden_dim, self.output_dim))
        return nn.ModuleList(layers)

    def __init__(self, net_params: Params) -> None:
        """_summary_"""
        super().__init__(net_params)
        self.layers = self._load_layers(net_params)

    def forward(self, embedding: th.Tensor, inst_feat: th.Tensor) -> th.Tensor:
        input = th.cat([embedding, inst_feat], dim=1)
        output = input
        for layer in self.layers:
            output = layer(output)
        return output
