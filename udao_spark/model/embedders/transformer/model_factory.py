"""This module contains functions to create graph transformers."""

from typing import Optional

import dgl
import torch as th
from torch import nn as nn

import udao_spark.model.embedders.transformer.graph_transformer as udao_graph_transformer  # noqa: E501
import udao_spark.model.embedders.transformer.layers.layer_factory as udao_layer_factory


def create_graph_transformer(
    in_dim: int,
    output_size: int,
    n_layers: int,
    n_heads: int,
    pos_encoding_dim: Optional[int],
    hist_embedding_dim: Optional[int],
    bitmap_embedding_dim: Optional[int],
    n_op_types: Optional[int],
    op_groups: list[str],
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
        type_embedding_dim (int): encoding dimension of type embedding
        op_groups (list[str]): list of attributes of node encodings
        n_op_types: (Optional[int]): input dimension of type embedding
        bitmap_embedding_dim (Optional[int]): dimension of bitmap embedding
        hist_embedding_dim (Optional[int]): dimension of hist embedding
        pos_encoding_dim (Optional[int]): dimension of positional encoding

    """
    # define feature extractor
    op_embedder = nn.Embedding(n_op_types, type_embedding_dim)  # type: ignore
    # TODO (glachaud): fix me (shouldn't the parameters be reversed?)
    op_hist_embedder: Optional[th.nn.Module] = None
    if hist_embedding_dim:
        op_hist_embedder = nn.Linear(150, hist_embedding_dim)
    op_bitmap_embedder: Optional[th.nn.Module] = None
    if bitmap_embedding_dim:
        op_bitmap_embedder = nn.Linear(1000, bitmap_embedding_dim)
    feature_extractor = udao_layer_factory.get_concatenate_features_layer(
        op_embedder, op_hist_embedder, op_bitmap_embedder, op_groups
    )

    # preprocessing layer of input features
    pre_processing_layers: list[udao_graph_transformer.PreProcessingLayer]
    # IMPORTANT: the dimension of type must now be added to the input dimension.
    if "type" in op_groups:
        in_dim += type_embedding_dim
    if "hist" in op_groups and hist_embedding_dim:
        in_dim += hist_embedding_dim
    if "bitmap" in op_groups and bitmap_embedding_dim:
        in_dim += bitmap_embedding_dim
    embedding_h = nn.Linear(in_dim, output_size)
    pre_processing_layers = [embedding_h]

    # positional encoding
    if pos_encoding_dim:
        embedding_lap_pos_enc = nn.Linear(pos_encoding_dim, output_size)
        positional_encoding_layer = udao_layer_factory.get_positional_encoding_layer(
            embedding_lap_pos_enc
        )
        pre_processing_layers.append(positional_encoding_layer)
        in_dim += pos_encoding_dim

    # functional for final readout to address pass the "h" of `mean_nodes`
    def final_readout(graph: udao_graph_transformer.Graph) -> th.Tensor:
        return dgl.mean_nodes(graph, "h")

    out_dims = []
    for idx_layer in range(n_layers):
        out_dims.append(output_size)
    layers: list[nn.Module] = []
    for idx, out_dim in enumerate(out_dims):
        layer = udao_layer_factory.get_multihead_attention_layer(
            in_dim=output_size if idx == 0 else out_dims[idx - 1],
            out_dim=out_dim,
            n_heads=n_heads,
            batch_norm=True,
        )
        layers.append(layer)
    graph_model = udao_graph_transformer.GraphTransformer(
        feature_extractor,
        preprocess_layers=pre_processing_layers,
        layers=layers,
        final_readout=final_readout,
    )
    return graph_model
