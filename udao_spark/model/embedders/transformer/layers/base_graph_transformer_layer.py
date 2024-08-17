"""This module contains the base class for Graph Transformer layers."""

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from udao_spark.model.embedders.transformer.graph_transformer import Graph


class BaseGraphTransformerLayer(th.nn.Module):
    """Base Graph Transformer Layer."""

    def __init__(
        self,
        attention_layer: nn.Module,
        in_dim: int,
        out_dim: int,
        n_heads: int = 1,
        dropout: float = 0.0,
        use_bias: bool = False,
        residual: bool = True,
        layer_norm: bool = False,
        batch_norm: bool = False,
    ):
        """Instantiate a BaseGraphTransformerLayer.

        Args:
            attention_layer (th.nn.Module): attention mechanism of the layer
            in_dim (int): Size of each input sample
            out_dim (int): Size of each output sample
            n_heads (int): Number of multi-head-attentions (default: `1`)
            dropout (float): Dropout probability (default: `0`)
            use_bias (bool): Whether to use bias or not (default: `False`)
            residual (bool): Whether to use residual connections
                or not (default: `True`)
            layer_norm (bool): Whether to apply layer normalization
                or not. Defaults to False.
            batch_norm (bool): Whether to apply batch normalization
                or not. Defaults to False.
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_heads = n_heads
        self.dropout = dropout
        self.use_bias = use_bias
        self.residual = residual
        self.attention = attention_layer
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm

        self.O = nn.Linear(out_dim, out_dim)

        if self.layer_norm:
            self.layer_norm1 = nn.LayerNorm(out_dim)

        if self.batch_norm:
            self.batch_norm1 = nn.BatchNorm1d(out_dim)

        self.FFN_layer1 = nn.Linear(out_dim, out_dim * 2)
        self.FFN_layer2 = nn.Linear(out_dim * 2, out_dim)

        if self.layer_norm:
            self.layer_norm2 = nn.LayerNorm(out_dim)

        if self.batch_norm:
            self.batch_norm2 = nn.BatchNorm1d(out_dim)

    def forward(self, graph: Graph, h: th.Tensor) -> th.Tensor:
        """Compute the forward pass of the base graph transformer layer.

        Args:
            graph (Graph): _description_
            h (th.Tensor): feature matrix of the nodes

        Returns:
            th.Tensor: tensor representation of the nodes
        """
        h_in1 = h  # for first residual connection

        # multi-head attention out
        attn_out = self.attention(graph, h)
        h = attn_out.view(-1, self.out_channels)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.O(h)

        if self.residual:
            h = h_in1 + h  # residual connection

        if self.layer_norm:
            h = self.layer_norm1(h)

        if self.batch_norm:
            h = self.batch_norm1(h)

        h_in2 = h  # for second residual connection

        # FFN
        h = self.FFN_layer1(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_layer2(h)

        if self.residual:
            h = h_in2 + h  # residual connection

        if self.layer_norm:
            h = self.layer_norm2(h)

        if self.batch_norm:
            h = self.batch_norm2(h)
        return h
