"""This module contains base class for graph-based query embedders."""

from typing import Callable, List, Optional, Union

import dgl
import torch as th
import torch_geometric
from torch import nn as nn

# type aliases to simplify the signature of the GraphTransformer
Graph = Union[dgl.DGLGraph, torch_geometric.data.Data]  # type: ignore
Readout = Callable[[Graph], th.Tensor]
GraphFeatureExtractor = Callable[[Graph], th.Tensor]


class GraphTransformer(th.nn.Module):
    """Base class for graph-based query embedders."""

    def __init__(
        self,
        feature_extractor: GraphFeatureExtractor,
        preprocess_layers: Optional[List[th.nn.Module]],
        layers: List[th.nn.Module],
        final_readout: Readout,
    ):
        """Instantiate a Graph Transformer for query graph embedding.

        Args:
            feature_extractor (GraphFeatureExtractor): function that retrieves
                the feature matrix of the input graph.
            preprocess_layers (Optional[List[th.nn.Module]]):
                pre-processing operations to perform on the input data.
            layers (List[th.nn.Module]):
                list of modules to apply in sequential order
            final_readout (Readout): operation that is applied after the
                layers, and which produces the final graph representation.
        """

        super().__init__()
        self.feature_extractor = feature_extractor

        self.preprocess_layers: list[nn.Module]
        if preprocess_layers:
            if isinstance(preprocess_layers, nn.Module):
                self.preprocess_layers = [preprocess_layers]
            self.preprocess_layers = preprocess_layers

        # Register layers as Modules
        self.layers = th.nn.ModuleList(layers)

        self.final_readout = final_readout

    def forward(self, graph: Graph) -> th.Tensor:
        """Compute the forward pass of the Graph Transformer.

        Args:
            graph (Graph): graph of a query plan
            h (th.Tensor): features matrix of the nodes

        Returns:
            th.Tensor: tensor representation of the query graph.
        """
        input_features = self.feature_extractor(graph)
        # apply preprocessing layers
        if self.preprocess_layers:
            for pre_process_layer in self.preprocess_layers:
                h = pre_process_layer(input_features)
        else:
            h = input_features

        # apply forward layers
        for layer in self.layers:
            graph = layer(graph, h)

        return self.final_readout(graph)
