import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch as th

from udao_trace.utils.logging import logger

from .base_optimizer import BaseOptimizer
from .utils import get_cloud_cost_add_io, get_cloud_cost_wo_io


class HierarchicalOptimizer(BaseOptimizer):
    def extract_non_decision_df(self, non_decision_input: Dict) -> pd.DataFrame:
        """
        compute the graph_embedding and
        the normalized values of the non-decision variables
        """
        df = pd.DataFrame.from_dict(non_decision_input, orient="index")
        df = df.reset_index().rename(columns={"index": "id"})
        df["id"] = df["id"].str.split("-").str[-1].astype(int)
        df.set_index("id", inplace=True, drop=False)
        df.sort_index(inplace=True)
        return df

    def get_objective_values_mlp(
        self,
        graph_embeddings: th.Tensor,
        non_decision_tabular_features: th.Tensor,
        theta: th.Tensor,
    ) -> Dict[str, np.ndarray]:
        tabular_features = th.cat([non_decision_tabular_features, theta], dim=1)
        objs = self._predict_objectives_mlp(graph_embeddings, tabular_features).numpy()
        obj_io = objs[:, 1]
        obj_ana_lat = objs[:, 2]
        theta_c_min, theta_c_max = self.theta_minmax["c"]
        k1_min, k2_min, k3_min = theta_c_min[:3]
        k1_max, k2_max, k3_max = theta_c_max[:3]
        k1 = (theta[:, 0].numpy() - k1_min) * (k1_max - k1_min) + k1_min
        k2 = (theta[:, 1].numpy() - k2_min) * (k2_max - k2_min) + k2_min
        k3 = (theta[:, 2].numpy() - k3_min) * (k3_max - k3_min) + k3_min
        return self._summarize_obj(k1, k2, k3, obj_ana_lat, obj_io)

    def _summarize_obj(
        self,
        k1: np.ndarray,
        k2: np.ndarray,
        k3: np.ndarray,
        obj_ana_lat: np.ndarray,
        obj_io: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        obj_ana_cost_wo_io = get_cloud_cost_wo_io(
            lat=obj_ana_lat,
            cores=k1,
            mem=k1 * k2,
            nexec=k3,
        )
        obj_ana_cost_w_io = get_cloud_cost_add_io(obj_ana_cost_wo_io, obj_io)
        if not isinstance(obj_ana_cost_wo_io, np.ndarray) or not isinstance(
            obj_ana_cost_w_io, np.ndarray
        ):
            raise TypeError(
                f"Expected np.ndarray, "
                f"got {type(obj_ana_cost_wo_io)} and {type(obj_ana_cost_w_io)}"
            )
        return {
            "ana_latency": obj_ana_lat,
            "io": obj_io,
            "ana_cost_wo_io": obj_ana_cost_wo_io,
            "ana_cost_w_io": obj_ana_cost_w_io,
        }

    def get_objective_values_ag(
        self,
        graph_embeddings: np.ndarray,
        non_decision_df: pd.DataFrame,
        sampled_theta: np.ndarray,
        model_name: str,
    ) -> Dict[str, np.ndarray]:
        start_time_ns = time.perf_counter_ns()
        objs = self.ag_ms.predict_with_ag(
            self.bm, graph_embeddings, non_decision_df, sampled_theta, model_name
        )
        end_time_ns = time.perf_counter_ns()
        logger.info(
            f"takes {(end_time_ns - start_time_ns) / 1e6} ms "
            f"to run {len(sampled_theta)} theta"
        )
        return self._summarize_obj(
            sampled_theta[:, 0],
            sampled_theta[:, 1],
            sampled_theta[:, 2],
            np.array(objs["ana_latency_s"]),
            np.array(objs["io_mb"]),
        )

    def foo_samples(
        self, n_stages: int, seed: Optional[int], normalize: bool
    ) -> np.ndarray:
        # a naive way to sample
        theta_c = np.tile(
            self.sample_theta_x(
                1, "c", seed if seed is not None else None, normalize=normalize
            ),
            (n_stages, 1),
        )
        theta_p = self.sample_theta_x(
            n_stages, "p", seed + 1 if seed is not None else None, normalize=normalize
        )
        theta_s = self.sample_theta_x(
            n_stages, "s", seed + 2 if seed is not None else None, normalize=normalize
        )
        theta = np.concatenate([theta_c, theta_p, theta_s], axis=1)
        return theta

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
        logger.info("graph_embeddings shape: %s", graph_embeddings.shape)
        logger.warning(
            "non_decision_tabular_features is only used for MLP inferencing, shape: %s",
            non_decision_tabular_features.shape,
        )

        n_stages = len(non_decision_input)
        if use_ag:
            graph_embeddings = graph_embeddings.detach().cpu()
            if graph_embeddings.shape[0] != n_stages:
                raise ValueError(
                    f"graph_embeddings shape {graph_embeddings.shape} "
                    f"does not match n_stages {n_stages}"
                )
            sampled_theta = self.foo_samples(n_stages, seed, normalize=False)
            if ag_model is None:
                logger.warning(
                    "ag_model is not specified, choosing the ensembled model"
                )
                ag_model = "WeightedEnsemble_L2"
            objs_dict = self.get_objective_values_ag(
                graph_embeddings.numpy(), non_decision_df, sampled_theta, ag_model
            )
        else:
            # use MLP for inference.
            sampled_theta = self.foo_samples(n_stages, seed, normalize=True)
            # an example to get objective values given theta
            objs_dict = self.get_objective_values_mlp(
                graph_embeddings,
                non_decision_tabular_features,
                th.tensor(sampled_theta, dtype=self.dtype),
            )
            logger.info(objs_dict)

        index = 0
        theta_chosen = sampled_theta[index]
        logger.info(theta_chosen)
        if use_ag:
            conf = self.sc.construct_configuration(
                theta_chosen.reshape(1, -1).astype(float)
            ).squeeze()
        else:
            conf = self.sc.construct_configuration_from_norm(
                theta_chosen.reshape(1, -1)
            ).squeeze()
        objs = np.array(
            [
                objs_dict["ana_latency"][index],
                objs_dict["ana_cost_w_io"][index],
            ]
        )

        logger.info(f"conf: {conf}")
        logger.info(f"objs: {objs}")
        return conf, objs
