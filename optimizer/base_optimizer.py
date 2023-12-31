from abc import ABC, abstractmethod
from typing import List

from udao.optimization.utils.moo_utils import Point


class BaseOptimizer(ABC):
    @abstractmethod
    def solve(self) -> List[Point]:
        pass
