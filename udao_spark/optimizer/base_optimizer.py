import os.path
from abc import ABC, abstractmethod
from typing import Dict, List

import torch as th
from udao.data.handler.data_processor import DataProcessor
from udao.optimization.utils.moo_utils import Point

from udao_trace.utils import PickleHandler

from ..model.model_server import ModelServer


class BaseOptimizer(ABC):
    def __init__(
        self,
        model_sign: str,
        model_params_path: str,
        weights_path: str,
        data_processor_path: str,
    ) -> None:
        self.ms = ModelServer.from_ckp_path(model_sign, model_params_path, weights_path)
        self.data_processor = PickleHandler.load(
            os.path.dirname(data_processor_path), os.path.basename(data_processor_path)
        )
        if not isinstance(self.data_processor, DataProcessor):
            raise TypeError(f"Expected DataHandler, got {type(self.data_processor)}")

    def weighted_utopia_nearest(self, pareto_points: List[Point]) -> Point:
        """
        return the Pareto point that is closest to the utopia point
        in a weighted distance function
        """
        # todo
        pass

    def predict_objectives(
        self, graph_embedding: th.Tensor, tabular_features: th.Tensor
    ) -> th.Tensor:
        """
        return multiple objective values for a given graph embedding and
        tabular features in the normalized space
        """
        return self.ms.model.regressor(graph_embedding, tabular_features)

    @abstractmethod
    def solve(self, non_decision_input: Dict) -> List[Point]:
        pass
