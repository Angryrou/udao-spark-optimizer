from typing import Any, List

from udao.optimization.utils.moo_utils import Point

from .base_optimizer import BaseOptimizer


class AtomicOptimizer(BaseOptimizer):
    def solve(self, non_decision_vars: List[float], **kwargs: Any) -> List[Point]:
        assert "graph_str" in kwargs and isinstance(kwargs["graph_str"], str)
        return []
