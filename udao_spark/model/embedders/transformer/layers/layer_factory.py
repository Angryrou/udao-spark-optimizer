"""This module contains functions to create graph transformer layers."""

from typing import Optional, Union

import dgl
import torch as th
import torch_geometric
from udao.model.embedders.layers.multi_head_attention import MultiHeadAttentionLayer

from udao_spark.model.embedders.transformer.graph_transformer import (
    GraphFeatureExtractor,
    PreProcessingLayer,
)
from udao_spark.model.embedders.transformer.layers import (
    base_graph_transformer_layer as udao_spark_embedders_base_layers,
)

Graph = Union[dgl.DGLGraph, torch_geometric.data.Data]  # type: ignore
BaseGraphTransformerType = udao_spark_embedders_base_layers.BaseGraphTransformerLayer


def get_multihead_attention_layer(
    in_dim: int,
    out_dim: int,
    n_heads: int = 1,
    dropout: float = 0.0,
    use_bias: bool = False,
    residual: bool = True,
    layer_norm: bool = False,
    batch_norm: bool = False,
) -> BaseGraphTransformerType:
    """Create a multi-head attention layer

    Args:
        in_dim (int): Size of each input sample
        out_dim (int): Size of each output sample
        n_heads (int, optional): Number of multi-head attentions. Defaults to 1.
        dropout (float, optional): Dropout probability. Defaults to 0.0.
        use_bias (bool, optional): Whether to use bias or not. Defaults to False.
        residual (bool, optional): Whether to use residual connections or not.
            Defaults to True.  # type: ignore
        layer_norm (bool, optional): Whether to apply layer normalization
            or not. Defaults to False.
        batch_norm (bool, optional): Whether to apply batch normalization
            or not. Defaults to False.
    """
    attention_layer = MultiHeadAttentionLayer(
        in_dim=in_dim, out_dim=out_dim // n_heads, n_heads=n_heads, use_bias=use_bias
    )
    return udao_spark_embedders_base_layers.BaseGraphTransformerLayer(
        attention_layer,
        in_dim,
        out_dim,
        n_heads,
        dropout,
        use_bias,
        residual,
        layer_norm,
        batch_norm,
    )


# TODO(glachaud): refactoring in progress
def get_concatenate_features_layer(
    op_embedder: th.nn.Module,
    op_hist_embedder: Optional[th.nn.Module],
    op_bitmap_embedder: Optional[th.nn.Module],
    op_groups: list[str],
) -> GraphFeatureExtractor:
    """Create a concatenation layer

    Args:
        op_embedder (th.nn.Module): Linear layer
        op_bitmap_embedder (Optional[th.nn.Module]): Linear layer
        op_hist_embedder (Optional[th.nn.Module]): Linear layer
        op_groups (list[str]): list of attributes of node encodings
    """

    # TODO (glachaud): refactoring in progress
    def concatenate_op_features(g: dgl.DGLGraph) -> th.Tensor:
        op_list = []
        if "type" in op_groups:
            op_list.append(op_embedder(g.ndata["op_gid"]))
        if "cbo" in op_groups:
            op_list.append(g.ndata["cbo"])
        if "op_enc" in op_groups:
            op_list.append(g.ndata["op_enc"])
        if "hist" in op_groups and op_hist_embedder:
            op_list.append(op_hist_embedder(g.ndata["hist"]))
        if "bitmap" in op_groups and op_bitmap_embedder:
            op_list.append(op_bitmap_embedder(g.ndata["bitmap"]))
        op_tensor = th.cat(op_list, dim=1) if len(op_list) > 1 else op_list[0]
        return op_tensor

    return concatenate_op_features


def get_positional_encoding_layer(
    positional_embedder: th.nn.Module,
) -> PreProcessingLayer:
    """
    Create a positional encoding layer.
    Parameters
    ----------
    positional_embedder: Linear layer

    Returns
    -------

    """

    def positional_encoding(g: Graph, h: th.Tensor) -> th.Tensor:
        # currently supports only DGL graph
        h_lap_pos_enc = positional_embedder(g.ndata["pos_enc"])
        h = h + h_lap_pos_enc
        return h

    return positional_encoding
