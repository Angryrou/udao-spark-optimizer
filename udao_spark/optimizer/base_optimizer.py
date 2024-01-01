import os.path
from abc import ABC

import torch as th
from udao.data.handler.data_handler import DataHandler

from udao_trace.utils import PickleHandler

from ..model.model_server import ModelServer


class BaseOptimizer(ABC):
    def __init__(
        self,
        model_sign: str,
        model_params_path: str,
        weights_path: str,
        data_handler_path: str,
    ) -> None:
        self.ms = ModelServer.from_ckp_path(model_sign, model_params_path, weights_path)
        self.data_handler = PickleHandler.load(
            os.path.dirname(data_handler_path), os.path.basename(data_handler_path)
        )
        if not isinstance(self.data_handler, DataHandler):
            raise TypeError(f"Expected DataHandler, got {type(self.data_handler)}")
        self.data_processor = self.data_handler.data_processor

    def predict_objectives(
        self, graph_embedding: th.Tensor, tabular_features: th.Tensor
    ) -> th.Tensor:
        """
        return multiple objective values for a given graph embedding and
        tabular features in the normalized space
        """
        return self.ms.model.regressor(graph_embedding, tabular_features)
