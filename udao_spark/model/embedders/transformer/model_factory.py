"""This module contains functions to create graph transformers."""

from typing import Optional

import dgl
import torch as th
from torch import nn as nn

import udao_spark.model.embedders.transformer.graph_transformer as udao_graph_transformer  # noqa: E501
import udao_spark.model.embedders.transformer.layers.layer_factory as udao_layer_factory


def create_graph_transformer(
    in_dim: int,
    hidden_dim: int,
    output_size: int,
    n_layers: int,
    n_heads: int,
    n_op_types: Optional[int],
    op_type: bool,
    op_cbo: bool,
    op_enc: bool,
    type_embedding_dim: int,
) -> udao_graph_transformer.GraphTransformer:
    """Create a Graph Transformer model.

    This model is a GTN without positional encoding.

    Args:
        in_dim (int): Size of each sample
        hidden_dim (int): Size of each layer
        output_size (int): Size of each output sample
        n_layers (int): Number of layers
        n_heads (int): Number of multi-head attentions.
    """
    # define feature extractor
    op_embedder = nn.Embedding(n_op_types, type_embedding_dim)  # type: ignore
    feature_extractor = udao_layer_factory.get_concatenate_features_layer(
        op_embedder, op_type=op_type, op_cbo=op_cbo, op_enc=op_enc
    )

    # preprocessing layer of input features
    # IMPORTANT: the dimension of type must now be added to the input dimension.
    if op_type:
        in_dim += type_embedding_dim
    embedding_h = nn.Linear(in_dim, hidden_dim)

    # functional for final readout to address pass the "h" of `mean_nodes`
    def final_readout(graph: udao_graph_transformer.Graph) -> th.Tensor:
        return dgl.mean_nodes(graph, "h")

    out_dims = []
    for idx_layer in range(n_layers):
        out_dims.append(hidden_dim if idx_layer < n_layers - 1 else output_size)
    layers: list[nn.Module] = []
    for out_dim in out_dims:
        layer = udao_layer_factory.get_multihead_attention_layer(
            in_dim=in_dim,
            out_dim=out_dim,
            n_heads=n_heads,
            batch_norm=True,
        )
        layers.append(layer)
    graph_model = udao_graph_transformer.GraphTransformer(
        feature_extractor,
        preprocess_layers=[embedding_h],
        layers=layers,
        final_readout=final_readout,
    )
    return graph_model
