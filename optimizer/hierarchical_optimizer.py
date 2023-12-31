from typing import Any, List

from udao.optimization.utils.moo_utils import Point

from .base_optimizer import BaseOptimizer


class HierarchicalOptimizer(BaseOptimizer):
    def solve(self, non_decision_vars: List[float], **kwargs: Any) -> List[Point]:
        assert "graph_strs" in kwargs and isinstance(kwargs["graph_strs"], List)
        return []
