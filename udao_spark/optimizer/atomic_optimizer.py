from typing import Dict, List

from udao.optimization.utils.moo_utils import Point

from .base_optimizer import BaseOptimizer


class AtomicOptimizer(BaseOptimizer):
    def solve(self, non_decision_input: Dict) -> List[Point]:
        return []
