import itertools
import time
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch as th

from udao_spark.optimizer.moo_algos.div_and_conq_moo import DivAndConqMOO
from udao_trace.utils.logging import logger

from .base_optimizer import BaseOptimizer
from .utils import get_cloud_cost_add_io, get_cloud_cost_wo_io, save_results


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

    def get_objective_values(
        self,
        graph_embeddings: th.Tensor,
        non_decision_tabular_features: th.Tensor,
        theta: th.Tensor,
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

        return th.vstack([obj_ana_lat, obj_ana_cost_w_io]).T
        # return th.vstack([obj_ana_lat, obj_io]).T

    def solve(
        self,
        non_decision_input: Dict,
        seed: Optional[int] = None,
        algo: Optional[str] = "div_and_conq_moo",
        sample_mode: Optional[str] = "random",
        query_id: Optional[str] = "-1",
        save_data: bool = False,
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

            index = 0
            theta_chosen = theta[index]
            logger.info(theta_chosen)
            conf = self.sc.construct_configuration_from_norm(
                theta_chosen.numpy().reshape(1, -1)
            ).squeeze()
        elif algo == "div_and_conq_moo":
            start = time.time()
            n_stages = len(non_decision_input)
            theta_s = self.sample_theta_x(
                1, "s", seed + 2 if seed is not None else None
            )
            if sample_mode == "random":
                theta_c = self.sample_theta_x(
                    1000, "c", seed if seed is not None else None
                )
                theta_p = self.sample_theta_x(
                    2000, "p", seed + 1 if seed is not None else None
                )
            elif sample_mode == "grid":
                c_grids = [
                    [1, 2],
                    [1, 2],
                    [4, 16],
                    # [5],
                    # [4],
                    # [16],
                    [1, 4],
                    [0, 5],
                    [0, 1],
                    [0, 1],
                    [50, 75],
                ]
                p_grids = [
                    [0, 5],
                    [1, 6],
                    [0, 32],
                    [0, 32],
                    [2, 50],
                    [0, 4],
                    [20, 80],
                    [0, 4],
                    [0, 4],
                ]
                # s_grids = [[5], [1]]
                c_samples = np.array([list(i) for i in itertools.product(*c_grids)])
                p_samples = np.array([list(i) for i in itertools.product(*p_grids)])
                # s_samples = np.array([list(i) for i in itertools.product(*s_grids)])
                c_samples_norm = (c_samples - self.theta_minmax["c"][0]) / (
                    self.theta_minmax["c"][1] - self.theta_minmax["c"][0]
                )
                theta_c = th.tensor(c_samples_norm, dtype=th.float32)
                p_samples_norm = (p_samples - self.theta_minmax["p"][0]) / (
                    self.theta_minmax["p"][1] - self.theta_minmax["p"][0]
                )
                theta_p = th.tensor(p_samples_norm, dtype=th.float32)
                # s_samples_norm = (s_samples - self.theta_minmax["s"][0]) / (
                #     self.theta_minmax["s"][1] - self.theta_minmax["s"][0]
                # )
                # theta_s = th.tensor(s_samples_norm, dtype=th.float32)

            else:
                raise Exception(
                    f"The sample mode {sample_mode} for theta is not supported!"
                )

            len_theta_per_qs = theta_c.shape[1] + theta_p.shape[1] + theta_s.shape[1]
            dag_opt_algo = "hier_moo%11"
            div_moo = DivAndConqMOO(
                n_stages=n_stages,
                graph_embeddings=graph_embeddings,
                non_decision_tabular_features=non_decision_tabular_features,
                obj_model=self.get_objective_values,
                params=DivAndConqMOO.Params(
                    c_samples=theta_c,
                    p_samples=theta_p,
                    s_samples=theta_s,
                    n_clusters=10,
                    cross_location=3,
                    dag_opt_algo=dag_opt_algo,
                ),
                seed=0,
            )
            po_objs, po_conf = div_moo.solve()
            time_cost = time.time() - start
            print(f"FUNCTION: time cost of div_and_conq_moo is: {time_cost}")
            print(
                f"The number of Pareto solutions in the DAG opt method"
                f" {dag_opt_algo} is: "
                f"{np.unique(po_objs, axis=0).shape[0]}"
            )
            # todo: apply WUN
            if save_data:
                conf_qs0 = po_conf[:, :len_theta_per_qs].reshape(-1, len_theta_per_qs)
                conf2 = self.sc.construct_configuration_from_norm(conf_qs0).squeeze()
                data_path = (
                    f"./output/test_{self.device.type}/div_and_conq_moo/time_-1/"
                    f"query_{query_id}_n_{n_stages}/{sample_mode}/{dag_opt_algo}"
                )
                save_results(data_path, po_objs, mode="F")
                save_results(data_path, conf2, mode="Theta")
                save_results(data_path, np.array([time_cost]), mode="time")

            objs, conf_norm = self.weighted_utopia_nearest(po_objs, po_conf)
            assert conf_norm.shape[0] == n_stages * len_theta_per_qs
            # conf of theta_c, along with theta_p and theta_s of qs0
            conf_norm_qs0 = conf_norm[:len_theta_per_qs].reshape(1, -1)
            conf = self.sc.construct_configuration_from_norm(conf_norm_qs0).squeeze()
            print(
                f"FUNCTION: time cost of div_and_conq_moo with WUN "
                f"is: {time.time() - start}"
            )

        else:
            raise Exception(
                f"Compile-time optimization algorithm {algo} is not supported!"
            )

        logger.info(f"conf: {conf}")
        logger.info(f"objs: {objs}")
        return conf, objs
