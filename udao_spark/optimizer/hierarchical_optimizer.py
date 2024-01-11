from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch as th

from udao_trace.utils.logging import logger

from .base_optimizer import BaseOptimizer
from .utils import get_cloud_cost_add_io, get_cloud_cost_wo_io


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

    # def get_objective_values(
    #     self,
    #     graph_embeddings: th.Tensor,
    #     non_decision_tabular_features: th.Tensor,
    #     theta: th.Tensor,
    #     mode: Optional[str] = "2D_general",
    # ) -> Dict[str, th.Tensor]:
    #     tabular_features = th.cat([non_decision_tabular_features, theta], dim=1)
    #     objs = self._predict_objectives(graph_embeddings, tabular_features)
    #     obj_io = objs[:, 1]
    #     obj_ana_lat = objs[:, 2]
    #     obj_ana_cost = self.get_cloud_cost(
    #         lat=obj_ana_lat,
    #         cores=theta[:, 0],
    #         mem=theta[:, 0] * theta[:, 1],
    #         nexec=theta[:, 2],
    #     )
    #
    #     if mode == "2D_general":
    #         return {
    #             "ana_latency": obj_ana_lat,
    #             "io": obj_io,
    #         }
    #     elif mode == "2D_simple":
    #         return {
    #             "ana_latency": obj_ana_lat,
    #             "ana_cost": obj_ana_cost,
    #         }
    #     else:
    #         return {
    #             "ana_latency": obj_ana_lat,
    #             "ana_cost": obj_ana_cost,
    #             "io": obj_io,
    #         }

    def get_objective_values(
        self,
        graph_embeddings: th.Tensor,
        non_decision_tabular_features: th.Tensor,
        theta: th.Tensor,
        mode: Optional[str] = "2D_general",
    ) -> th.Tensor:
        tabular_features = th.cat([non_decision_tabular_features, theta], dim=1)
        objs = self._predict_objectives(graph_embeddings, tabular_features)
        obj_io = objs[:, 1]
        obj_ana_lat = objs[:, 2]
        obj_ana_cost_wo_io = get_cloud_cost_wo_io(
            lat=obj_ana_lat,
            cores=theta[:, 0],
            mem=theta[:, 0] * theta[:, 1],
            nexec=theta[:, 2],
        )
        obj_ana_cost_w_io = get_cloud_cost_add_io(obj_ana_cost_wo_io, obj_io)
        if not isinstance(obj_ana_cost_wo_io, th.Tensor) or not isinstance(
            obj_ana_cost_w_io, th.Tensor
        ):
            raise TypeError(
                f"Expected th.Tensor, "
                f"got {type(obj_ana_cost_wo_io)} and {type(obj_ana_cost_w_io)}"
            )

        return obj_ana_cost_w_io
        # return {
        #     "ana_latency": obj_ana_lat,
        #     "io": obj_io,
        #     "ana_cost_wo_io": obj_ana_cost_wo_io,
        #     "ana_cost_w_io": obj_ana_cost_w_io,
        # }

    def solve(
        self,
        non_decision_input: Dict,
        seed: Optional[int] = None,
        algo: Optional[str] = "div_and_conq_moo",
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
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
        if algo == "naive_sample":
            # a naive way to sample
            n_stages = len(non_decision_input)
            theta_c = self.sample_theta_x(
                1, "c", seed if seed is not None else None
            ).repeat(n_stages, 1)
            theta_p = self.sample_theta_x(
                n_stages, "p", seed + 1 if seed is not None else None
            )
            theta_s = self.sample_theta_x(
                n_stages, "s", seed + 2 if seed is not None else None
            )
            theta = th.cat([theta_c, theta_p, theta_s], dim=1)

            # an example to get objective values given theta
            objs_tensor = self.get_objective_values(
                graph_embeddings, non_decision_tabular_features, theta
            )
            objs = objs_tensor.detach().numpy()
            logger.info(objs)

        # an example to get objective values given theta
        objs_dict = self.get_objective_values(
            graph_embeddings, non_decision_tabular_features, theta
        )
        logger.info(objs_dict)

        index = 0
        theta_chosen = theta[index]
        logger.info(theta_chosen)
        conf = self.sc.construct_configuration_from_norm(
            theta_chosen.numpy().reshape(1, -1)
        ).squeeze()
        # objs = np.array(
        #     [
        #         objs_dict["ana_latency"][index],
        #         objs_dict["ana_cost_w_io"][index],
        #     ]
        # )
        objs = objs_dict.numpy()

        logger.info(f"conf: {conf}")
        logger.info(f"objs: {objs}")
        return conf, objs
