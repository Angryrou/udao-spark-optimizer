from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch as th

from udao_spark.optimizer.moo_algos.div_and_conq_moo import DivAndConqMOO
from udao_trace.utils.logging import logger

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
        obj_ana_cost = self.get_cloud_cost(
            lat=obj_ana_lat,
            cores=theta[:, 0],
            mem=theta[:, 0] * theta[:, 1],
            nexec=theta[:, 2],
        )

        if mode == "2D_general":
            return th.vstack([obj_ana_lat, obj_io]).T
        elif mode == "2D_simple":
            return th.vstack([obj_ana_lat, obj_ana_cost]).T
        else:
            return th.vstack([obj_ana_lat, obj_ana_cost, obj_io]).T

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

            # fixme: the final returned theta should have 8 + (9 + 2) * n_stages
            index = 0
            theta_chosen = theta[index]
            logger.info(theta_chosen)
            conf = self.sc.construct_configuration_from_norm(
                theta_chosen.numpy().reshape(1, -1)
            ).squeeze()

        elif algo == "div_and_conq_moo":
            n_stages = len(non_decision_input)
            theta_c = self.sample_theta_x(2000, "c", seed if seed is not None else None)
            theta_p = self.sample_theta_x(
                2000, "p", seed + 1 if seed is not None else None
            )
            theta_s = self.sample_theta_x(
                1, "s", seed + 2 if seed is not None else None
            )

            div_moo = DivAndConqMOO(
                n_stages=n_stages,
                graph_embeddings=graph_embeddings,
                non_decision_tabular_features=non_decision_tabular_features,
                obj_model=self.get_objective_values,
                params=DivAndConqMOO.Params(
                    c_samples=theta_c,
                    p_samples=theta_p,
                    s_samples=theta_s,
                    n_clusters=100,
                    cross_location=3,
                    dag_opt_algo="approx_solve",
                ),
                seed=0,
            )
            po_objs, po_conf = div_moo.solve()

            # todo: apply WUN
            objs, conf_norm = self.weighted_utopia_nearest(po_objs, po_conf)
            conf = self.sc.construct_configuration_from_norm(
                conf_norm.reshape(1, -1)
            ).squeeze()

        else:
            raise Exception(
                f"Compile-time optimization algorithm {algo} is not supported!"
            )

        logger.info(f"conf: {conf}")
        logger.info(f"objs: {objs}")
        return conf, objs
