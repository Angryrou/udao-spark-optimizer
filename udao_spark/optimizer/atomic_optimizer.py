from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .base_optimizer import BaseOptimizer


class AtomicOptimizer(BaseOptimizer):
    def extract_non_decision_df(self, non_decision_input: Dict) -> pd.DataFrame:
        """
        extract the non_decision dict to a DataFrame
        """
        df = pd.DataFrame.from_dict({0: non_decision_input}, orient="index")
        df.index.name = "id"
        return df

    def solve(
        self,
        non_decision_input: Dict[str, Any],
        seed: Optional[int] = None,
        use_ag: bool = True,
        ag_model: Optional[str] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        non_decision_df = self.extract_non_decision_df(non_decision_input)
        (
            graph_embeddings,
            non_decision_tabular_features,
        ) = self.extract_non_decision_embeddings_from_df(non_decision_df)

        return None, None
