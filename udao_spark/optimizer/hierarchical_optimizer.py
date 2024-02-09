import copy
import itertools
import time
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pygmo as pg  # type: ignore
import torch as th
from udao.optimization.concepts import BoolVariable, FloatVariable, IntegerVariable

from udao_spark.optimizer.base_optimizer import BaseOptimizer
from udao_spark.optimizer.moo_algos.div_and_conq_moo import DivAndConqMOO
from udao_spark.optimizer.moo_algos.evo_optimizer import EvoOptimizer
from udao_spark.optimizer.moo_algos.ws_optimizer import WSOptimizer
from udao_spark.optimizer.utils import (
    even_weights,
    get_cloud_cost_add_io,
    get_cloud_cost_wo_io,
    save_results,
)
from udao_trace.utils.interface import VarTypes
from udao_trace.utils.logging import logger


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

    def get_objective_values_mlp_arr(
        self,
        graph_embeddings: th.Tensor,
        non_decision_tabular_features: th.Tensor,
        theta: th.Tensor,
        place: str = "",
    ) -> np.ndarray:
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
        objs_dict = self._summarize_obj(k1, k2, k3, obj_ana_lat, obj_io)

        obj_cost_w_io = objs_dict["ana_cost_w_io"]

        return np.vstack((obj_ana_lat, obj_cost_w_io)).T

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
        objs = self.ag_ms.predict_with_ag(
            self.bm, graph_embeddings, non_decision_df, sampled_theta, model_name
        )
        return self._summarize_obj(
            sampled_theta[:, 0],
            sampled_theta[:, 1],
            sampled_theta[:, 2],
            np.array(objs["ana_latency_s"]),
            np.array(objs["io_mb"]),
        )

    def get_objective_values_ag_arr(
        self,
        graph_embeddings: np.ndarray,
        non_decision_df: pd.DataFrame,
        sampled_theta: np.ndarray,
        model_name: str,
    ) -> np.ndarray:
        start_time_ns = time.perf_counter_ns()
        objs = self.ag_ms.predict_with_ag(
            self.bm, graph_embeddings, non_decision_df, sampled_theta, model_name
        )
        end_time_ns = time.perf_counter_ns()
        logger.info(
            f"takes {(end_time_ns - start_time_ns) / 1e6} ms "
            f"to run {len(sampled_theta)} theta"
        )
        objs_dict = self._summarize_obj(
            sampled_theta[:, 0],
            sampled_theta[:, 1],
            sampled_theta[:, 2],
            np.array(objs["ana_latency_s"]),
            np.array(objs["io_mb"]),
        )

        ana_latency = objs["ana_latency_s"]
        ana_cost_w_io = objs_dict["ana_cost_w_io"]

        return np.vstack((ana_latency, ana_cost_w_io)).T

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
        non_decision_input: Dict,
        seed: Optional[int] = None,
        use_ag: bool = True,
        ag_model: str = "",
        algo: str = "naive_example",
        save_data: bool = False,
        query_id: Optional[str] = None,
        sample_mode: Optional[str] = None,
        param1: int = -1,
        param2: int = -1,
        time_limit: int = -1,
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
        len_theta_c = len(self.theta_ktype["c"])
        len_theta_p = len(self.theta_ktype["p"])
        len_theta_s = len(self.theta_ktype["s"])
        len_theta_per_qs = len_theta_c + len_theta_p + len_theta_s

        theta_c: Union[th.Tensor, np.ndarray]
        theta_p: Union[th.Tensor, np.ndarray]
        theta_s: Union[th.Tensor, np.ndarray]

        start = time.time()

        if algo == "naive_example":
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
                # objs_dict = self.get_objective_values_ag(
                #     graph_embeddings.numpy(), non_decision_df, sampled_theta, ag_model
                # )
                objs = self.get_objective_values_ag_arr(
                    graph_embeddings.numpy(), non_decision_df, sampled_theta, ag_model
                )
            else:
                # use MLP for inference.
                sampled_theta = self.foo_samples(n_stages, seed, normalize=True)
                # an example to get objective values given theta
                # objs_dict = self.get_objective_values_mlp(
                #     graph_embeddings,
                #     non_decision_tabular_features,
                #     th.tensor(sampled_theta, dtype=self.dtype),
                # )
                # logger.info(objs_dict)
                objs = self.get_objective_values_mlp_arr(
                    graph_embeddings,
                    non_decision_tabular_features,
                    th.tensor(sampled_theta, dtype=self.dtype),
                )

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
            # objs = np.array(
            #     [
            #         objs_dict["ana_latency"][index],
            #         objs_dict["ana_cost_w_io"][index],
            #     ]
            # )

            logger.info(f"conf: {conf}")
            logger.info(f"objs: {objs}")
            print(f"Query {query_id} finished!")
            return conf, objs

        elif algo == "model_inference_time":
            fake_objs = np.array([-1])
            start_infer = time.time()
            # n_repeat = 10
            for n_repeat in [10, 100, 1000, 10000]:
                if use_ag:
                    graph_embeddings = graph_embeddings.detach().cpu()
                    if graph_embeddings.shape[0] != n_stages:
                        raise ValueError(
                            f"graph_embeddings shape {graph_embeddings.shape} "
                            f"does not match n_stages {n_stages}"
                        )
                    normalize = False
                    theta_c = np.tile(
                        self.sample_theta_x(
                            1,
                            "c",
                            seed if seed is not None else None,
                            normalize=normalize,
                        ),
                        (n_stages * n_repeat, 1),
                    )
                    theta_p = self.sample_theta_x(
                        n_stages * n_repeat,
                        "p",
                        seed + 1 if seed is not None else None,
                        normalize=normalize,
                    )
                    theta_s = self.sample_theta_x(
                        n_stages * n_repeat,
                        "s",
                        seed + 2 if seed is not None else None,
                        normalize=normalize,
                    )
                    theta = np.concatenate([theta_c, theta_p, theta_s], axis=1)
                    if ag_model is None:
                        logger.warning(
                            "ag_model is not specified, choosing the ensembled model"
                        )
                        ag_model = "WeightedEnsemble_L2"
                    mesh_graph_embeddings = th.repeat_interleave(
                        graph_embeddings, n_repeat, dim=0
                    )
                    mesh_non_decision_df = non_decision_df.loc[
                        np.repeat(non_decision_df.index, n_repeat)
                    ].reset_index(drop=True)
                    assert (
                        mesh_graph_embeddings.shape[0] == mesh_non_decision_df.shape[0]
                    )
                    assert mesh_graph_embeddings.shape[0] == theta.shape[0]

                    tc_list_pure_pred = []
                    for i in range(5):
                        start_pred = time.time()
                        objs = self.get_objective_values_ag_arr(
                            mesh_graph_embeddings.numpy(),
                            mesh_non_decision_df,
                            theta,
                            ag_model,
                        )
                        time_cost_pred = time.time() - start_pred
                        tc_list_pure_pred.append(time_cost_pred)

                    if save_data:
                        data_path = (
                            f"./output/updated_model_{self.device.type}/"
                            f"test/{algo}_update/time_-1/"
                            f"query_{query_id}_n_{n_stages}/"
                            f"n_rows_{n_repeat * n_stages}"
                        )
                        save_results(
                            data_path, np.array(tc_list_pure_pred), mode="time"
                        )

                else:
                    # use MLP for inference.
                    sampled_theta = self.foo_samples(n_stages, seed, normalize=True)
                    objs = self.get_objective_values_mlp_arr(
                        graph_embeddings,
                        non_decision_tabular_features,
                        th.tensor(sampled_theta, dtype=self.dtype),
                    )

                time_cost = time.time() - start_infer
                print(
                    f"time cost of model prediction (including setting theta) "
                    f"with {n_repeat * n_stages} rows is {time_cost}"
                )

            return fake_objs, fake_objs

        elif algo == "analyze_model_accuracy":
            theta_s = self.sample_theta_x(
                1,
                "s",
                seed + 2 if seed is not None else None,
                normalize=False,
            )

            c_grids = [
                # [1, 5],
                # [1, 4],
                # [4, 16],
                # [1, 4],
                # [0, 5],
                # [0, 1],
                # [0, 1],
                # [50, 75],
                [1],
                [1],
                [4],
                [1],
                [0],
                [0],
                [0],
                [50],
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

            test_theta_c = np.array(sum(c_grids, [])).reshape(-1, len_theta_c)
            test_theta_p = np.array(np.array(p_grids)[:, 0].tolist()).reshape(
                -1, len_theta_p
            )
            test_theta = np.concatenate(
                [
                    test_theta_c,
                    test_theta_p,
                    theta_s,
                ],
                axis=1,
            ).repeat(n_stages, axis=0)
            assert test_theta.shape[0] == graph_embeddings.shape[0]
            assert test_theta.shape[0] == non_decision_df.shape[0]
            test_objs = self.get_objective_values_ag_arr(
                graph_embeddings.numpy(), non_decision_df, test_theta, ag_model
            )
            test_query_objs = test_objs.sum(0)
            if save_data:
                conf_qs0 = test_theta[0, :len_theta_per_qs].reshape(1, len_theta_per_qs)
                conf2 = self.sc.construct_configuration(
                    conf_qs0.reshape(1, -1).astype(float)
                ).squeeze()
                data_path = (
                    f"./output/updated_model_{self.device.type}/{algo}/time_-1/"
                    f"query_{query_id}_n_{n_stages}/c_{conf2[0]}_{conf2[1]}_{conf2[2]}/grid"
                )
                save_results(data_path, test_query_objs, mode="F")
                save_results(data_path, conf2, mode="Theta")

            return test_query_objs, test_query_objs

        elif "analyze_multi_control" in algo:
            n_samples_p = 100
            n_repeat_samples = 10  # 1, 2, 5, 10, 100
            normalize = False
            # e.g. algo = "analyze_multi_control%1"
            theta_s = self.sample_theta_x(
                1, "s", seed + 2 if seed is not None else None, normalize=normalize
            )
            if algo.split("%")[1] == "1":
                mesh_theta_s_sub_control = theta_s.repeat(
                    n_samples_p * n_repeat_samples * n_stages, axis=0
                )
            else:
                mesh_theta_s_sub_control = np.concatenate(
                    [
                        self.sample_theta_x(
                            n_samples_p * n_repeat_samples,
                            "s",
                            seed + 4 + x if seed is not None else None,
                            normalize=normalize,
                        )
                        for x in range(n_stages)
                    ],
                    axis=0,
                )

            theta_c = self.sample_theta_x(
                n_samples_p,
                "c",
                seed if seed is not None else None,
                normalize=normalize,
            )
            theta_p_query_control = self.sample_theta_x(
                n_samples_p,
                "p",
                seed + 1 if seed is not None else None,
                normalize=normalize,
            )

            start_query_control = time.time()
            n_repeat_query_control = theta_p_query_control.shape[0]
            # mesh_theta_c_query_control = theta_c.repeat(n_repeat_query_control, 1)
            mesh_theta_c_query_control = theta_c
            mesh_theta_s_query_control = theta_s.repeat(n_repeat_query_control, axis=0)
            theta_query_control = np.concatenate(
                [
                    mesh_theta_c_query_control,
                    theta_p_query_control,
                    mesh_theta_s_query_control,
                ],
                axis=1,
            )
            mesh_theta_query_control = theta_query_control.repeat(n_stages, axis=0)
            mesh_graph_embeddings_query_control = graph_embeddings.repeat_interleave(
                n_repeat_query_control, dim=0
            )
            mesh_non_decision_df_query_control = non_decision_df.loc[
                np.repeat(non_decision_df.index, n_repeat_query_control)
            ].reset_index(drop=True)
            objs_query_control = self.get_objective_values_ag_arr(
                mesh_graph_embeddings_query_control.numpy(),
                mesh_non_decision_df_query_control,
                mesh_theta_query_control,
                ag_model,
            )

            # shape (n_repeat, n_stages)
            latency_query_control = np.vstack(
                np.split(objs_query_control[:, 0], n_stages)
            ).T
            cost_query_control = np.vstack(
                np.split(objs_query_control[:, 1], n_stages)
            ).T

            query_latency_query_control = np.sum(latency_query_control, axis=1)
            query_cost_query_control = np.sum(cost_query_control, axis=1)
            query_objs_query_control = np.vstack(
                (query_latency_query_control, query_cost_query_control)
            ).T
            assert query_objs_query_control.shape[0] == n_repeat_query_control

            po_query_ind_query_control = pg.non_dominated_front_2d(
                query_objs_query_control
            )
            po_query_objs_query_control = query_objs_query_control[
                po_query_ind_query_control
            ]
            po_query_confs_qs0_query_control = theta_query_control[
                po_query_ind_query_control
            ]
            time_cost_query_control = time.time() - start_query_control

            start_sub_control = time.time()
            theta_p_sub_control_all = np.concatenate(
                [
                    self.sample_theta_x(
                        n_samples_p * n_repeat_samples,
                        "p",
                        seed + 3 + x if seed is not None else None,
                        normalize=normalize,
                    )
                    for x in range(n_stages)
                ],
                axis=0,
            )
            assert theta_p_sub_control_all.shape[1] == len_theta_p
            mesh_theta_c_sub_control = theta_c.repeat(n_repeat_samples, axis=0).repeat(
                n_stages, axis=0
            )
            # mesh_theta_s_sub_control = theta_s.repeat(n_repeat_sub_control, 1)
            mesh_theta_sub_control = np.concatenate(
                [
                    mesh_theta_c_sub_control,
                    theta_p_sub_control_all,
                    mesh_theta_s_sub_control,
                ],
                axis=1,
            )
            mesh_graph_embeddings_sub_control = graph_embeddings.repeat_interleave(
                n_samples_p * n_repeat_samples, dim=0
            )
            mesh_non_decision_df_sub_control = non_decision_df.loc[
                np.repeat(non_decision_df.index, n_samples_p * n_repeat_samples)
            ].reset_index(drop=True)
            objs_sub_control = self.get_objective_values_ag_arr(
                mesh_graph_embeddings_sub_control.numpy(),
                mesh_non_decision_df_sub_control,
                mesh_theta_sub_control,
                ag_model,
            )

            # shape (n_repeat, n_stages)
            latency_sub_control = np.vstack(
                np.split(objs_sub_control[:, 0], n_stages)
            ).T
            cost_sub_control = np.vstack(np.split(objs_sub_control[:, 1], n_stages)).T

            query_latency_sub_control = np.sum(latency_sub_control, axis=1)
            query_cost_sub_control = np.sum(cost_sub_control, axis=1)
            query_objs_sub_control = np.vstack(
                (query_latency_sub_control, query_cost_sub_control)
            ).T

            po_query_ind_sub_control = pg.non_dominated_front_2d(query_objs_sub_control)
            po_query_objs_sub_control = query_objs_sub_control[po_query_ind_sub_control]
            po_query_confs_qs0_sub_control = mesh_theta_sub_control[
                : n_samples_p * n_repeat_samples, :
            ][po_query_ind_sub_control]

            time_cost_sub_control = time.time() - start_sub_control

            if save_data:
                data_path_query_control = (
                    f"./output/updated_model_{self.device.type}/expr1_multi_control/{algo}/"
                    f"query_{query_id}_n_{n_stages}/query_control/{n_samples_p}_{n_repeat_samples}"
                )
                data_path_sub_control = (
                    f"./output/updated_model_{self.device.type}/expr1_multi_control/{algo}/"
                    f"query_{query_id}_n_{n_stages}/sub_control/{n_samples_p}_{n_repeat_samples}"
                )
                conf_query_control = self.sc.construct_configuration(
                    po_query_confs_qs0_query_control.astype(float)
                ).reshape(-1, len_theta_per_qs)
                conf_sub_control = self.sc.construct_configuration(
                    po_query_confs_qs0_sub_control.astype(float)
                ).reshape(-1, len_theta_per_qs)

                save_results(
                    data_path_query_control, po_query_objs_query_control, mode="F"
                )
                save_results(data_path_query_control, conf_query_control, mode="Theta")
                save_results(
                    data_path_query_control,
                    np.array([time_cost_query_control]),
                    mode="time",
                )

                save_results(data_path_sub_control, po_query_objs_sub_control, mode="F")
                save_results(data_path_sub_control, conf_sub_control, mode="Theta")
                save_results(
                    data_path_sub_control,
                    np.array([time_cost_sub_control]),
                    mode="time",
                )

            return po_query_objs_sub_control, conf_sub_control

        elif "div_and_conq_moo" in algo:
            # algo = div_and_conq_moo%GD
            dag_opt_algo = algo.split("%")[1]

            if use_ag:
                normalize = False
            else:
                normalize = True
            theta_s_samples = self.sample_theta_x(
                1, "s", seed + 2 if seed is not None else None, normalize=normalize
            )
            if use_ag:
                theta_s = theta_s_samples
            else:
                theta_s = th.tensor(theta_s_samples, dtype=th.float32)

            if sample_mode == "random":
                theta_c_samples = self.sample_theta_x(
                    10, "c", seed if seed is not None else None, normalize=normalize
                )
                theta_p_samples = self.sample_theta_x(
                    10, "p", seed + 1 if seed is not None else None, normalize=normalize
                )
                if use_ag:
                    theta_c = theta_c_samples
                    theta_p = theta_p_samples
                else:
                    theta_c = th.tensor(theta_c_samples, dtype=th.float32)
                    theta_p = th.tensor(theta_p_samples, dtype=th.float32)
            elif sample_mode == "grid":
                c_grids = [
                    [1, 5],
                    [1, 4],
                    [4, 16],
                    [1, 4],
                    [0, 5],
                    [0, 1],
                    [0, 1],
                    [50, 75],
                    # [1],
                    # [1],
                    # [16],
                    # [4],  # random1: 3, random2: 4
                    # [2],  # random1: 3, random2: 2
                    # [1],  # random1: 1
                    # [1],  # random1: 1
                    # [70],  # random1: 60, random2: 70
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

                if use_ag:
                    theta_c = np.array([list(i) for i in itertools.product(*c_grids)])
                    theta_p = np.array([list(i) for i in itertools.product(*p_grids)])
                else:
                    theta_c_samples = np.array(
                        [list(i) for i in itertools.product(*c_grids)]
                    )
                    theta_p_samples = np.array(
                        [list(i) for i in itertools.product(*p_grids)]
                    )
                    # s_samples = np.array([list(i)
                    # for i in itertools.product(*s_grids)])
                    c_samples_norm = (theta_c_samples - self.theta_minmax["c"][0]) / (
                        self.theta_minmax["c"][1] - self.theta_minmax["c"][0]
                    )
                    theta_c = th.tensor(c_samples_norm, dtype=th.float32)
                    p_samples_norm = (theta_p_samples - self.theta_minmax["p"][0]) / (
                        self.theta_minmax["p"][1] - self.theta_minmax["p"][0]
                    )
                    theta_p = th.tensor(p_samples_norm, dtype=th.float32)

            else:
                raise Exception(
                    f"The sample mode {sample_mode} for theta is not supported!"
                )

            # len_theta_per_qs = theta_c.shape[1] + theta_p.shape[1] + theta_s.shape[1]
            n_clusters = 10 if theta_c.shape[0] > 10 else theta_c.shape[0]

            non_decision_features: Union[th.Tensor, pd.DataFrame]
            obj_model: Union[
                Callable[[th.Tensor, th.Tensor, th.Tensor, str], np.ndarray],
                Callable[[np.ndarray, pd.DataFrame, np.ndarray, str], np.ndarray],
            ]
            if use_ag:
                obj_model = self.get_objective_values_ag_arr
                non_decision_features = non_decision_df
            else:
                obj_model = self.get_objective_values_mlp_arr
                non_decision_features = non_decision_tabular_features

            div_moo = DivAndConqMOO(
                n_stages=n_stages,
                graph_embeddings=graph_embeddings,
                non_decision_tabular_features=non_decision_features,
                obj_model=obj_model,
                ag_model=ag_model,
                use_ag=use_ag,
                params=DivAndConqMOO.Params(
                    c_samples=theta_c,
                    p_samples=theta_p,
                    s_samples=theta_s,
                    n_clusters=n_clusters,
                    cross_location=3,
                    dag_opt_algo=dag_opt_algo,
                    verbose=True,
                ),
                seed=0,
            )
            po_objs, po_conf = div_moo.solve()
            time_cost = time.time() - start
            print(f"query id is {query_id}")
            print(f"FUNCTION: time cost of div_and_conq_moo is: {time_cost}")
            print(
                f"The number of Pareto solutions in the DAG opt method"
                f" {dag_opt_algo} is: "
                f"{np.unique(po_objs, axis=0).shape[0]}"
            )

            conf_qs0 = po_conf[:, :len_theta_per_qs].reshape(-1, len_theta_per_qs)
            if use_ag:
                conf2 = self.sc.construct_configuration(conf_qs0).reshape(
                    -1, len_theta_per_qs
                )
                data_path = (
                    f"./output/test/check_bottleneck/updated_model_{self.device.type}/ag/{algo}/time_-1/"
                    f"query_{query_id}_n_{n_stages}/{sample_mode}/use_ag_{use_ag}/{dag_opt_algo}"
                )
            else:
                conf2 = self.sc.construct_configuration_from_norm(conf_qs0).reshape(
                    -1, len_theta_per_qs
                )
                data_path = (
                    f"./output/test/check_bottleneck/updated_model_{self.device.type}/mlp/{algo}/time_-1/"
                    f"query_{query_id}_n_{n_stages}/{sample_mode}/use_ag_{use_ag}/{dag_opt_algo}"
                )

            if save_data:
                save_results(data_path, po_objs, mode="F")
                save_results(data_path, conf2, mode="Theta")
                save_results(data_path, np.array([time_cost]), mode="time")

            # add WUN
            objs, conf = self.weighted_utopia_nearest(po_objs, conf2)
            print(
                f"FUNCTION: time cost of {algo} with WUN " f"is: {time.time() - start}"
            )

            logger.info(f"conf: {conf}")
            logger.info(f"objs: {objs}")
            return conf, objs
            # return po_objs, po_conf

        elif algo == "evo":
            if use_ag:
                normalize = False
            else:
                normalize = True
            theta_s_samples = self.sample_theta_x(
                1, "s", seed + 2 if seed is not None else None, normalize=normalize
            )
            if use_ag:
                theta_s = theta_s_samples
                theta_minmax = copy.deepcopy(self.theta_minmax)
                theta_minmax["s"][0][:] = theta_s[0]
                theta_minmax["s"][1][:] = theta_s[0]
                theta_type = self.theta_ktype
            else:
                theta_s = th.tensor(theta_s_samples, dtype=th.float32)

                theta_minmax = {
                    k: (
                        np.zeros_like(v[0]).astype(float),
                        np.ones_like(v[1]).astype(float),
                    )
                    for k, v in self.theta_minmax.items()
                }
                theta_minmax["s"][0][:] = theta_s[0]
                theta_minmax["s"][1][:] = theta_s[0]
                theta_type = {
                    k: [VarTypes.FLOAT] * len(v) for k, v in self.theta_ktype.items()
                }

            if use_ag:
                obj_model = self.get_objective_values_ag_arr
                non_decision_features = non_decision_df
            else:
                obj_model = self.get_objective_values_mlp_arr
                non_decision_features = non_decision_tabular_features
            # pop_size=10
            # nfe=10000
            pop_size = param1
            nfe = param2
            time_limit = time_limit
            evo = EvoOptimizer(
                query_id=query_id,
                n_stages=n_stages,
                graph_embeddings=graph_embeddings,
                non_decision_tabular_features=non_decision_features,
                obj_model=obj_model,
                params=EvoOptimizer.Params(
                    pop_size=pop_size,
                    nfe=nfe,
                    fix_randomness_flag=True,
                    time_limit=time_limit,
                ),
                use_ag=use_ag,
                ag_model=ag_model,
                theta_minmax=theta_minmax,
                theta_ktype=theta_type,
            )
            po_objs, po_conf = evo.solve()
            time_cost = time.time() - start
            print(f"FUNCTION: time cost of div_and_conq_moo is: {time_cost}")
            print(
                f"The number of Pareto solutions in the {algo} "
                f"is {np.unique(po_objs, axis=0).shape[0]}"
            )
            print()

            if -1 in po_objs:  # time out
                po_objs = np.array(po_objs)
                conf2 = np.array(po_conf)
            else:
                conf_qs0 = po_conf[:, :len_theta_per_qs].reshape(-1, len_theta_per_qs)
                if use_ag:
                    conf2 = self.sc.construct_configuration(
                        conf_qs0.astype(float)
                    ).reshape(-1, len_theta_per_qs)

                else:
                    conf2 = self.sc.construct_configuration_from_norm(conf_qs0).reshape(
                        -1, len_theta_per_qs
                    )

            if use_ag:
                data_path = (
                    f"./output/test/check_bottleneck/updated_model_{self.device.type}/ag/{algo}/{pop_size}_{nfe}/time_{time_limit}/"
                    f"query_{query_id}_n_{n_stages}/"
                )
            else:
                data_path = (
                    f"./output/test/check_bottleneck/updated_model_{self.device.type}/mlp/{algo}/{pop_size}_{nfe}/time_{time_limit}/"
                    f"query_{query_id}_n_{n_stages}/"
                )

            if save_data:
                save_results(data_path, po_objs, mode="F")
                save_results(data_path, conf2, mode="Theta")
                save_results(data_path, np.array([time_cost]), mode="time")

            # add WUN
            objs, conf = self.weighted_utopia_nearest(po_objs, conf2)
            print(
                f"FUNCTION: time cost of {algo} with WUN " f"is: {time.time() - start}"
            )
            logger.info(f"conf: {conf}")
            logger.info(f"objs: {objs}")
            return conf, objs

        elif algo == "ws":
            if use_ag:
                normalize = False
            else:
                normalize = True
            theta_s_samples = self.sample_theta_x(
                1, "s", seed + 2 if seed is not None else None, normalize=normalize
            )

            if use_ag:
                theta_s = theta_s_samples
                c_vars = [
                    IntegerVariable(1, 5),
                    IntegerVariable(1, 4),
                    IntegerVariable(4, 16),
                    IntegerVariable(1, 4),
                    IntegerVariable(0, 5),
                    BoolVariable(),
                    BoolVariable(),
                    IntegerVariable(50, 75),
                ]
                p_vars = [
                    IntegerVariable(0, 5),
                    IntegerVariable(1, 6),
                    IntegerVariable(0, 32),
                    IntegerVariable(0, 32),
                    IntegerVariable(2, 50),
                    IntegerVariable(0, 4),
                    IntegerVariable(20, 80),
                    IntegerVariable(0, 4),
                    IntegerVariable(0, 4),
                ]
                s_vars = [IntegerVariable(x, x) for x in theta_s[0].tolist()]
            else:
                theta_s = th.tensor(theta_s_samples, dtype=th.float32)
                c_vars = [FloatVariable(0, 1)] * len(self.theta_ktype["c"])
                p_vars = [FloatVariable(0, 1)] * len(self.theta_ktype["p"])
                s_vars = [FloatVariable(x, x) for x in theta_s[0].numpy().tolist()]

            if use_ag:
                obj_model = self.get_objective_values_ag_arr
                non_decision_features = non_decision_df
            else:
                obj_model = self.get_objective_values_mlp_arr
                non_decision_features = non_decision_tabular_features
            # n_samples_per_param=100
            # n_ws = 11
            n_samples_per_param = param1
            n_ws = param2
            time_limit = time_limit
            # weights
            n_objs = 2
            ws_steps = 1 / (int(n_ws) - 1)
            ws_pairs = even_weights(ws_steps, n_objs)
            ws = WSOptimizer(
                query_id=query_id,
                n_stages=n_stages,
                graph_embeddings=graph_embeddings,
                non_decision_tabular_features=non_decision_features,
                obj_model=obj_model,
                params=WSOptimizer.Params(
                    n_samples_per_param=n_samples_per_param,
                    ws_pairs=ws_pairs,
                    time_limit=time_limit,
                ),
                c_vars=c_vars,
                p_vars=p_vars,
                s_vars=s_vars,
                use_ag=use_ag,
                ag_model=ag_model,
            )
            po_objs, po_conf = ws.solve()
            time_cost = time.time() - start
            print(f"FUNCTION: time cost of {algo} is: {time_cost}")
            print(
                f"The number of Pareto solutions in the {algo} "
                f"is {np.unique(po_objs, axis=0).shape[0]}"
            )
            print()

            if -1 in po_objs:  # time out
                po_objs = np.array(po_objs)
                conf2 = np.array(po_conf)
            else:
                conf_qs0 = po_conf[:, :len_theta_per_qs].reshape(-1, len_theta_per_qs)
                if use_ag:
                    conf2 = self.sc.construct_configuration(
                        conf_qs0.astype(float)
                    ).reshape(-1, len_theta_per_qs)

                else:
                    conf2 = self.sc.construct_configuration_from_norm(conf_qs0).reshape(
                        -1, len_theta_per_qs
                    )

            if use_ag:
                data_path = (
                    f"./output/test/check_bottleneck/updated_model_{self.device.type}/ag/{algo}/{n_samples_per_param}_{n_ws}/time_{time_limit}/"
                    f"query_{query_id}_n_{n_stages}/"
                )
            else:
                data_path = (
                    f"./output/test/check_bottleneck/updated_model_{self.device.type}/mlp/{algo}/{n_samples_per_param}_{n_ws}/time_{time_limit}/"
                    f"query_{query_id}_n_{n_stages}/"
                )

            if save_data:
                save_results(data_path, po_objs, mode="F")
                save_results(data_path, conf2, mode="Theta")
                save_results(data_path, np.array([time_cost]), mode="time")

            # add WUN
            objs, conf = self.weighted_utopia_nearest(po_objs, conf2)
            print(
                f"FUNCTION: time cost of {algo} with WUN " f"is: {time.time() - start}"
            )
            logger.info(f"conf: {conf}")
            logger.info(f"objs: {objs}")
            return conf, objs

        else:
            raise Exception(
                f"Compile-time optimization algorithm {algo} is not supported!"
            )
