from typing import Any, Dict, Optional, Tuple

import numpy as np

from .base_optimizer import BaseOptimizer


class AtomicOptimizer(BaseOptimizer):
    def solve(
        self, non_decision_input: Dict[str, Any], seed: Optional[int] = None
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        return None, None
