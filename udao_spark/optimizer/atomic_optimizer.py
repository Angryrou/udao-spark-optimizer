from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch as th

from udao_trace.utils.logging import logger

from .base_optimizer import BaseOptimizer


class AtomicOptimizer(BaseOptimizer):
    def extract_non_decision_df(self, non_decision_input: Dict) -> pd.DataFrame:
        """
        extract the non_decision dict to a DataFrame
        """
        df = pd.DataFrame.from_dict({0: non_decision_input}, orient="index")
        df.index.name = "id"
        df["id"] = df.index
        return df

    def solve(
        self,
        non_decision_input: Dict[str, Any],
        seed: Optional[int] = None,
        use_ag: bool = True,
        ag_model: Optional[str] = None,
        sample_mode: str = "random_sample",
        n_samples: int = 1,
        moo_mode: str = "BF",
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        non_decision_df = self.extract_non_decision_df(non_decision_input)
        (
            graph_embeddings,
            non_decision_tabular_features,
        ) = self.extract_non_decision_embeddings_from_df(non_decision_df)
        logger.info("graph_embeddings shape: %s", graph_embeddings.shape)
        logger.warning(
            "non_decision_tabular_features is only used for MLP inferencing, shape: %s",
            non_decision_tabular_features.shape,
        )
        if sample_mode == "random":
            sampled_theta = self.sample_theta_all(
                n_samples=n_samples, seed=seed, normalize=not use_ag
            )[:, -len(self.decision_variables) :]
        elif sample_mode == "grid":
            raise NotImplementedError
        else:
            raise ValueError(f"sample_mode {sample_mode} is not supported")

        if use_ag:
            if ag_model is None:
                logger.warning(
                    "ag_model is not specified, choosing the ensembled model"
                )
                ag_model = "WeightedEnsemble_L2"
            graph_embeddings = graph_embeddings.detach().cpu()
            objs_dict = self.get_objective_values_ag(
                graph_embeddings.tile(n_samples, 1).numpy(),
                pd.DataFrame(
                    np.tile(non_decision_df.values, (n_samples, 1)),
                    columns=non_decision_df.columns,
                ),
                sampled_theta,
                ag_model,
            )
        else:
            objs_dict = self.get_objective_values_mlp(
                graph_embeddings.tile(n_samples, 1),
                non_decision_tabular_features.tile(n_samples, 1),
                th.tensor(sampled_theta, dtype=self.dtype),
            )
        print(objs_dict)
        return None, None
