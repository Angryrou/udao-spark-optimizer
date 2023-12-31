from abc import ABC, abstractmethod
from typing import Any, List

from udao.data.handler.data_handler import DataHandler
from udao.optimization.utils.moo_utils import Point

from model_server.model import ModelServer


class BaseOptimizer(ABC):
    def __init__(self, ms: ModelServer, dh: DataHandler) -> None:
        self.ms = ms
        self.dh = dh

    @abstractmethod
    def solve(self, non_decision_vars: List[float], **kwargs: Any) -> List[Point]:
        # Implementation of the solve method in the base class
        pass
