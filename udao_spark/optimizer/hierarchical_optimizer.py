from typing import Dict, List, Tuple

import pandas as pd
import torch as th
from udao.optimization.utils.moo_utils import Point

from udao_trace.utils.logging import logger

from ..utils.constants import THETA_C, THETA_P, THETA_S
from .base_optimizer import BaseOptimizer


class HierarchicalOptimizer(BaseOptimizer):
    def extract_non_decision_embeddings(
        self, non_decision_input: Dict
    ) -> Tuple[th.Tensor, th.Tensor]:
        """
        compute the graph_embedding and
        the normalized values of the non-decision variables
        """
        df = pd.DataFrame.from_dict(non_decision_input, orient="index")
        df = df.reset_index().rename(columns={"index": "id"})
        df["id"] = df["id"].str.split("-").str[-1].astype(int)
        df.set_index("id", inplace=True, drop=False)
        df.sort_index(inplace=True)
        return self.extract_non_decision_embeddings_from_df(df)

    def solve(self, non_decision_input: Dict) -> List[Point]:
        (
            graph_embeddings,
            non_decision_tabular_features,
        ) = self.extract_non_decision_embeddings(non_decision_input)
        logger.info("graph_embeddings shape: %s", graph_embeddings.shape)
        logger.info(
            "non_decision_tabular_features shape: %s",
            non_decision_tabular_features.shape,
        )

        # TODO: implement the hierarchical optimizer to get
        #  theta_c, theta_p, theta_s within [0-1] to feed the model

        # an exmaple to get objective values given theta_c, theta_p, theta_s
        n_stages = len(non_decision_input)
        theta_c = th.tensor([0.5] * len(THETA_C), device=self.device)
        theta_p_list = th.tensor(
            [[0.5] * len(THETA_P) for i in range(n_stages)], device=self.device
        )
        theta_s_list = th.tensor(
            [[0.5] * len(THETA_S) for i in range(n_stages)], device=self.device
        )
        # repeat theta_c bc it is shared among all stages
        thetas = th.cat(
            [theta_c.repeat(n_stages, 1), theta_p_list, theta_s_list], dim=1
        )
        tabular_features = th.cat([non_decision_tabular_features, thetas], dim=1)
        objs = self.predict_objectives(graph_embeddings, tabular_features)

        logger.info(objs)

        return []
