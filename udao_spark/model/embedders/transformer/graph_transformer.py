"""This module contains base class for graph-based query embedders."""
from typing import Callable, List, Optional, Union

import dgl
import torch as th
import torch_geometric

# type aliases to simplify the signature of the GraphTransformer
Graph = Union[dgl.DGLGraph, torch_geometric.data.Data]
Readout = Callable[[Graph], th.Tensor]


class GraphTransformer(th.nn.Module):
    """Base class for graph-based query embedders."""

    def __init__(
        self,
        preprocess_layers: Optional[List[th.nn.Module]],
        layers: List[th.nn.Module],
        final_readout: Readout,
    ):
        """Instantiate a Graph Transformer for query graph embedding.

        Args:
            preprocess_layers (Optional[List[th.nn.Module]]):
                pre-processing operations to perform on the input data.
            layers (List[th.nn.Module]):
                list of modules to apply in sequential order
            final_readout (Readout): operation that is applied after the
                layers, and which produces the final graph representation.
        """
        super().__init__()
        self.preprocess_layers = preprocess_layers

        # Register layers as Modules
        self.layers = th.nn.ModuleList(layers)

        self.final_readout = final_readout

    def forward(self, input: Graph) -> th.Tensor:
        """Compute the forward pass of the Graph Transformer.

        Args:
            inputs (Graph): graph of a query plan

        Returns:
            th.Tensor: tensor representation of the query graph.
        """
        # apply preprocessing layers
        if self.preprocess_layers:
            for pre_process_layer in self.preprocess_layers:
                input = pre_process_layer(input)

        # apply forward layers
        for layer in self.layers:
            input = layer(input)

        return self.final_readout(input)
