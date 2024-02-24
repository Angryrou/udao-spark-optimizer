import copy
import itertools
import time
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch as th
from udao.optimization.concepts import BoolVariable, FloatVariable, IntegerVariable

from udao_trace.utils.interface import VarTypes

from ..utils.logging import logger
from ..utils.monitor import DivAndConqMonitor, UdaoMonitor
from .base_optimizer import BaseOptimizer
from .moo_algos.div_and_conq_moo import DivAndConqMOO
from .moo_algos.evo_optimizer import EvoOptimizer
from .moo_algos.ws_optimizer import WSOptimizer
from .utils import even_weights, save_results, weighted_utopia_nearest_impl


class HierarchicalOptimizer(BaseOptimizer):
    def extract_non_decision_df(self, non_decision_input: Dict) -> pd.DataFrame:
        """
        extract the non_decision dict to a DataFrame
        """
        df = pd.DataFrame.from_dict(non_decision_input, orient="index")
        df = df.reset_index().rename(columns={"index": "id"})
        df["id"] = df["id"].str.split("-").str[-1].astype(int)
        df.set_index("id", inplace=True, drop=False)
        df.sort_index(inplace=True)
        return df

    def get_objective_values_mlp_arr(
        self,
        graph_embeddings: th.Tensor,
        non_decision_tabular_features: th.Tensor,
        theta: th.Tensor,
        place: str = "",
    ) -> np.ndarray:
        tabular_features = th.cat([non_decision_tabular_features, theta], dim=1)
        objs = self.predict_objectives_mlp(graph_embeddings, tabular_features).numpy()
        obj_io = objs[:, 1]
        obj_ana_lat = objs[:, 2]
        theta_c_min, theta_c_max = self.theta_minmax["c"]
        k1_min, k2_min, k3_min = theta_c_min[:3]
        k1_max, k2_max, k3_max = theta_c_max[:3]
        k1 = (theta[:, 0].numpy() - k1_min) * (k1_max - k1_min) + k1_min
        k2 = (theta[:, 1].numpy() - k2_min) * (k2_max - k2_min) + k2_min
        k3 = (theta[:, 2].numpy() - k3_min) * (k3_max - k3_min) + k3_min
        objs_dict = self.summarize_obj(k1, k2, k3, obj_ana_lat, obj_io)

        obj_cost_w_io = objs_dict["ana_cost_w_io"]

        return np.vstack((obj_ana_lat, obj_cost_w_io)).T

    def get_objective_values_ag_arr(
        self,
        graph_embeddings: np.ndarray,
        non_decision_df: pd.DataFrame,
        sampled_theta: np.ndarray,
        model_name: Dict[str, str],
        cost_choice: str = "ana_cost_w_io",
    ) -> np.ndarray:
        start_time_ns = time.perf_counter_ns()
        objs = self.ag_ms.predict_with_ag(
            self.bm,
            self.current_target_template,
            graph_embeddings,
            non_decision_df,
            self.decision_variables,
            sampled_theta,
            model_name,
        )
        end_time_ns = time.perf_counter_ns()
        logger.info(
            f"takes {(end_time_ns - start_time_ns) / 1e6} ms "
            f"to run {len(sampled_theta)} theta"
        )
        objs_dict = self.summarize_obj(
            sampled_theta[:, 0],
            sampled_theta[:, 1],
            sampled_theta[:, 2],
            np.array(objs["ana_latency_s"]),
            np.array(objs["io_mb"]),
        )

        ana_latency = objs["ana_latency_s"]
        ana_cost_w_io = objs_dict[cost_choice]

        return np.vstack((ana_latency, ana_cost_w_io)).T

    def solve(
        self,
        template: str,
        non_decision_input: Dict[str, Any],
        seed: Optional[int] = None,
        use_ag: bool = True,
        ag_model: Dict = dict(),
        algo: str = "naive_example",
        save_data: bool = False,
        query_id: Optional[str] = None,
        sample_mode: Optional[str] = None,
        param1: int = -1,
        param2: int = -1,
        time_limit: int = -1,
        is_oracle: bool = False,
        save_data_header: str = "./output",
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        self.current_target_template = template

        # initial a monitor
        monitor = DivAndConqMonitor() if "div_and_conq" in algo else UdaoMonitor()
        non_decision_dict = self.extract_data_and_compute_non_decision_features(
            monitor, non_decision_input
        )
        non_decision_df = non_decision_dict["non_decision_df"]
        graph_embeddings = non_decision_dict["graph_embeddings"]
        non_decision_tabular_features = non_decision_dict[
            "non_decision_tabular_features"
        ]
        # checking the monitor is working:
        if (
            monitor.input_extraction_ms <= 0
            or monitor.graph_embedding_computation_ms <= 0
        ):
            raise Exception("Monitor is not working properly!")
        logger.info(
            "input_extraction: {}ms, graph_embedding_computation: {}ms".format(
                monitor.input_extraction_ms, monitor.graph_embedding_computation_ms
            )
        )

        n_stages = len(non_decision_input)
        len_theta_c = len(self.theta_ktype["c"])
        len_theta_p = len(self.theta_ktype["p"])
        len_theta_s = len(self.theta_ktype["s"])
        len_theta_per_qs = len_theta_c + len_theta_p + len_theta_s

        if algo == "naive_example":
            objs, conf = self._naive_example(
                use_ag,
                graph_embeddings,
                n_stages,
                ag_model,
                non_decision_df,
                non_decision_tabular_features,
                seed,
            )

        elif algo == "multi_inference_time":
            objs, conf = self._model_inference_time(
                use_ag,
                graph_embeddings,
                n_stages,
                ag_model,
                non_decision_df,
                non_decision_tabular_features,
                seed,
                algo,
                save_data,
                query_id,
                save_data_header=save_data_header,
            )

        elif algo == "analyze_model_accuracy":
            objs, conf = self._model_accuracy(
                len_theta_p,
                len_theta_c,
                len_theta_per_qs,
                graph_embeddings,
                non_decision_df,
                non_decision_tabular_features,
                n_stages,
                seed,
                use_ag,
                ag_model,
                algo,
                query_id,
                save_data,
                save_data_header=save_data_header,
            )

        elif "div_and_conq_moo" in algo:
            # n_c_samples: param1
            # n_p_samples: param2
            objs, conf = self._div_and_conq_moo(
                len_theta_per_qs,
                graph_embeddings,
                non_decision_df,
                non_decision_tabular_features,
                n_stages,
                seed,
                use_ag,
                ag_model,
                algo,
                query_id,
                save_data,
                sample_mode,
                param1,
                param2,
                is_oracle,
                save_data_header,
            )

        elif algo == "evo":
            # pop_size = param1
            # nfe = param2
            objs, conf = self._evo(
                len_theta_per_qs,
                graph_embeddings,
                non_decision_df,
                non_decision_tabular_features,
                n_stages,
                seed,
                use_ag,
                ag_model,
                algo,
                query_id,
                save_data,
                param1,
                param2,
                time_limit,
                save_data_header,
            )

        elif algo == "ws":
            # n_samples_per_param = param1
            # n_ws = param2
            objs, conf = self._ws(
                len_theta_per_qs,
                graph_embeddings,
                non_decision_df,
                non_decision_tabular_features,
                n_stages,
                seed,
                use_ag,
                ag_model,
                algo,
                query_id,
                save_data,
                param1,
                param2,
                time_limit,
                save_data_header,
            )

        else:
            raise Exception(f"Algorithm {algo} is not supported!")

        # to save monitor:
        monitor.save_to_file(
            file_path=f"{save_data_header}/monitor/{self.bm}_{query_id}_{algo}_add_customized_params.json"
        )

        logger.info(f"conf: {conf}")
        logger.info(f"objs: {objs}")
        return objs, conf

    def _naive_example(
        self,
        use_ag: bool,
        graph_embeddings: th.Tensor,
        n_stages: int,
        ag_model: Dict[str, str],
        non_decision_df: pd.DataFrame,
        non_decision_tabular_features: th.Tensor,
        seed: Optional[int],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if use_ag:
            graph_embeddings = graph_embeddings.detach().cpu()
            if graph_embeddings.shape[0] != n_stages:
                raise ValueError(
                    f"graph_embeddings shape {graph_embeddings.shape} "
                    f"does not match n_stages {n_stages}"
                )
            sampled_theta = self.foo_samples(n_stages, seed, normalize=False)
            objs_dict = self.get_objective_values_ag(
                self.current_target_template,
                graph_embeddings.numpy(),
                non_decision_df,
                sampled_theta,
                ag_model,
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

        return objs, conf

    def _model_inference_time(
        self,
        use_ag: bool,
        graph_embeddings: th.Tensor,
        n_stages: int,
        ag_model: Dict[str, str],
        non_decision_df: pd.DataFrame,
        non_decision_tabular_features: th.Tensor,
        seed: Optional[int],
        algo: str,
        save_data: bool = False,
        query_id: Optional[str] = None,
        save_data_header: str = "./output",
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        fake_objs = np.array([-1])
        start_infer = time.time()
        theta_c: Union[th.Tensor, np.ndarray]
        theta_p: Union[th.Tensor, np.ndarray]
        theta_s: Union[th.Tensor, np.ndarray]
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
                mesh_graph_embeddings = th.repeat_interleave(
                    graph_embeddings, n_repeat, dim=0
                )
                mesh_non_decision_df = non_decision_df.loc[
                    np.repeat(non_decision_df.index, n_repeat)
                ].reset_index(drop=True)
                assert mesh_graph_embeddings.shape[0] == mesh_non_decision_df.shape[0]
                assert mesh_graph_embeddings.shape[0] == theta.shape[0]

                tc_list_pure_pred = []
                for i in range(5):
                    start_pred = time.time()
                    self.get_objective_values_ag_arr(
                        mesh_graph_embeddings.numpy(),
                        mesh_non_decision_df,
                        theta,
                        ag_model,
                    )
                    time_cost_pred = time.time() - start_pred
                    tc_list_pure_pred.append(time_cost_pred)

                if save_data:
                    data_path = (
                        f"{save_data_header}/latest_model_{self.device.type}/"
                        f"test/{algo}_update/time_-1/"
                        f"query_{query_id}_n_{n_stages}/"
                        f"n_rows_{n_repeat * n_stages}"
                    )
                    save_results(data_path, np.array(tc_list_pure_pred), mode="time")

            else:
                # use MLP for inference.
                sampled_theta = self.foo_samples(n_stages, seed, normalize=True)
                self.get_objective_values_mlp_arr(
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

    def _model_accuracy(
        self,
        len_theta_p: int,
        len_theta_c: int,
        len_theta_per_qs: int,
        graph_embeddings: th.Tensor,
        non_decision_df: pd.DataFrame,
        non_decision_tabular_features: th.Tensor,
        n_stages: int,
        seed: Optional[int],
        use_ag: bool,
        ag_model: Dict[str, str],
        algo: str,
        query_id: Optional[str],
        save_data: bool,
        save_data_header: str,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        theta_s: Union[th.Tensor, np.ndarray]
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
        assert test_theta.shape[0] == non_decision_tabular_features.shape[0]

        if use_ag:
            test_objs = self.get_objective_values_ag_arr(
                graph_embeddings.numpy(),
                non_decision_df,
                test_theta,
                ag_model,
            )
        else:
            test_objs = self.get_objective_values_mlp_arr(
                graph_embeddings.numpy(), non_decision_tabular_features, test_theta, ""
            )
        test_query_objs = test_objs.sum(0)
        if save_data:
            conf_qs0 = test_theta[0, :len_theta_per_qs].reshape(1, len_theta_per_qs)
            conf2 = self.sc.construct_configuration(
                conf_qs0.reshape(1, -1).astype(float)
            ).squeeze()
            data_path = (
                f"{save_data_header}/latest_model_{self.device.type}/{algo}/time_-1/"
                f"query_{query_id}_n_{n_stages}/c_{conf2[0]}_{conf2[1]}_{conf2[2]}/grid"
            )
            save_results(data_path, test_query_objs, mode="F")
            save_results(data_path, conf2, mode="Theta")

        return test_query_objs, test_query_objs

    def _div_and_conq_moo(
        self,
        len_theta_per_qs: int,
        graph_embeddings: th.Tensor,
        non_decision_df: pd.DataFrame,
        non_decision_tabular_features: th.Tensor,
        n_stages: int,
        seed: Optional[int],
        use_ag: bool,
        ag_model: Dict[str, str],
        algo: str,
        query_id: Optional[str],
        save_data: bool,
        sample_mode: Optional[str],
        n_c_samples: int,
        n_p_samples: int,
        is_oracle: bool,
        save_data_header: str,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        start = time.time()
        theta_c: Union[th.Tensor, np.ndarray]
        theta_p: Union[th.Tensor, np.ndarray]
        theta_s: Union[th.Tensor, np.ndarray]
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
                n_c_samples,
                "c",
                seed if seed is not None else None,
                normalize=normalize,
            )
            theta_p_samples = self.sample_theta_x(
                n_p_samples,
                "p",
                seed + 1 if seed is not None else None,
                normalize=normalize,
            )
            if use_ag:
                theta_c = theta_c_samples
                theta_p = theta_p_samples
            else:
                theta_c = th.tensor(theta_c_samples, dtype=th.float32)
                theta_p = th.tensor(theta_p_samples, dtype=th.float32)
        elif sample_mode == "grid":
            if n_c_samples == 256:
                c_grids = [
                    [1, 5],
                    [1, 4],
                    [4, 16],
                    [1, 4],
                    [0, 5],
                    [0, 1],
                    [0, 1],
                    [50, 75],
                ]
            elif n_c_samples == 128:
                c_grids = [
                    [1, 5],
                    [1, 4],
                    [4, 16],
                    [1, 4],
                    [0, 5],
                    [0, 1],
                    [1],
                    [50, 75],
                ]

            elif n_c_samples == 64:
                c_grids = [
                    [1, 5],
                    [1, 4],
                    [4, 16],
                    [1, 4],
                    [0, 5],
                    [0],
                    [1],
                    [50, 75],
                ]
            elif n_c_samples == 32:
                c_grids = [
                    [1, 5],
                    [1, 4],
                    [4, 16],
                    [1, 4],
                    [5],
                    [0],
                    [1],
                    [50, 75],
                ]
            elif n_c_samples == 16:
                c_grids = [
                    [1, 5],
                    [1, 4],
                    [4, 16],
                    [1, 4],
                    [5],
                    [0],
                    [1],
                    [50],
                ]
            else:
                raise Exception(f"n_c_samples {n_c_samples} is not supported!")

            if n_p_samples == 512:
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
            elif n_p_samples == 256:
                p_grids = [
                    [0],
                    [1, 6],
                    [0, 32],
                    [0, 32],
                    [2, 50],
                    [0, 4],
                    [20, 80],
                    [0, 4],
                    [0, 4],
                ]
            elif n_p_samples == 128:
                p_grids = [
                    [0],
                    [1, 6],
                    [0, 32],
                    [0, 32],
                    [2, 50],
                    [0, 4],
                    [20, 80],
                    [0],
                    [0, 4],
                ]
            elif n_p_samples == 64:
                p_grids = [
                    [0],
                    [1],
                    [0, 32],
                    [0, 32],
                    [2, 50],
                    [0, 4],
                    [20, 80],
                    [0],
                    [0, 4],
                ]
            elif n_p_samples == 32:
                p_grids = [
                    [0],
                    [1],
                    [0, 32],
                    [0, 32],
                    [2, 50],
                    [0],
                    [20, 80],
                    [0],
                    [0, 4],
                ]
            elif n_p_samples == 16:
                p_grids = [
                    [0],
                    [1],
                    [0, 32],
                    [0, 32],
                    [2, 50],
                    [0],
                    [20, 80],
                    [0],
                    [0],
                ]
            else:
                raise Exception(f"n_p_samples {n_p_samples} is not supported!")

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
                c_samples_norm = (theta_c_samples - self.theta_minmax["c"][0]) / (
                    self.theta_minmax["c"][1] - self.theta_minmax["c"][0]
                )
                theta_c = th.tensor(c_samples_norm, dtype=th.float32)
                p_samples_norm = (theta_p_samples - self.theta_minmax["p"][0]) / (
                    self.theta_minmax["p"][1] - self.theta_minmax["p"][0]
                )
                theta_p = th.tensor(p_samples_norm, dtype=th.float32)

            n_c_samples = theta_c.shape[0]
            n_p_samples = theta_p.shape[0]

        else:
            raise Exception(
                f"The sample mode {sample_mode} for theta is not supported!"
            )

        # len_theta_per_qs = theta_c.shape[1] + theta_p.shape[1] + theta_s.shape[1]
        n_clusters = 10 if theta_c.shape[0] > 10 else theta_c.shape[0]

        non_decision_features: Union[th.Tensor, pd.DataFrame]
        obj_model: Union[
            Callable[
                [np.ndarray, pd.DataFrame, np.ndarray, Dict[str, str]], np.ndarray
            ],
            Callable[[th.Tensor, th.Tensor, th.Tensor, str], np.ndarray],
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
            sample_funcs=self.sample_theta_x,
        )
        po_objs, po_conf, model_infer_info = div_moo.solve()
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
            conf_raw = self.sc.construct_configuration(conf_qs0).reshape(
                -1, len_theta_per_qs
            )
            data_path = (
                f"{save_data_header}/latest_model_{self.device.type}/ag/oracle_{is_oracle}/{algo}/{n_c_samples}_{n_p_samples}/time_-1/"
                f"query_{query_id}_n_{n_stages}/{sample_mode}/{dag_opt_algo}"
            )
        else:
            conf_raw = self.sc.construct_configuration_from_norm(conf_qs0).reshape(
                -1, len_theta_per_qs
            )
            data_path = (
                f"{save_data_header}/latest_model_{self.device.type}/mlp/oracle_{is_oracle}/{algo}/{n_c_samples}_{n_p_samples}/time_-1/"
                f"query_{query_id}_n_{n_stages}/{sample_mode}/{dag_opt_algo}"
            )

        if save_data:
            save_results(data_path, po_objs, mode="F")
            save_results(data_path, conf_raw, mode="Theta")
            save_results(data_path, np.array([time_cost]), mode="time")
            save_results(data_path, model_infer_info, mode="model_infer_info")

        # add WUN
        objs, conf = weighted_utopia_nearest_impl(po_objs, conf_raw)
        print(f"FUNCTION: time cost of {algo} with WUN " f"is: {time.time() - start}")

        return conf, objs

    def _evo(
        self,
        len_theta_per_qs: int,
        graph_embeddings: th.Tensor,
        non_decision_df: pd.DataFrame,
        non_decision_tabular_features: th.Tensor,
        n_stages: int,
        seed: Optional[int],
        use_ag: bool,
        ag_model: Dict[str, str],
        algo: str,
        query_id: Optional[str],
        save_data: bool,
        pop_size: int,
        nfe: int,
        time_limit: int,
        save_data_header: str,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        start = time.time()
        if use_ag:
            normalize = False
        else:
            normalize = True
        theta_s_samples = self.sample_theta_x(
            1, "s", seed + 2 if seed is not None else None, normalize=normalize
        )
        theta_s: Union[th.Tensor, np.ndarray]
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
        non_decision_features: Union[th.Tensor, pd.DataFrame]
        obj_model: Union[
            Callable[
                [np.ndarray, pd.DataFrame, np.ndarray, Dict[str, str]], np.ndarray
            ],
            Callable[[th.Tensor, th.Tensor, th.Tensor, str], np.ndarray],
        ]
        if use_ag:
            obj_model = self.get_objective_values_ag_arr
            non_decision_features = non_decision_df
        else:
            obj_model = self.get_objective_values_mlp_arr
            non_decision_features = non_decision_tabular_features

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
                conf2 = self.sc.construct_configuration(conf_qs0.astype(float)).reshape(
                    -1, len_theta_per_qs
                )

            else:
                conf2 = self.sc.construct_configuration_from_norm(conf_qs0).reshape(
                    -1, len_theta_per_qs
                )

        if use_ag:
            data_path = (
                f"{save_data_header}/latest_model_{self.device.type}/ag/{algo}/{pop_size}_{nfe}/time_{time_limit}/"
                f"query_{query_id}_n_{n_stages}/"
            )
        else:
            data_path = (
                f"{save_data_header}/latest_model_{self.device.type}/mlp/{algo}/{pop_size}_{nfe}/time_{time_limit}/"
                f"query_{query_id}_n_{n_stages}/"
            )

        if save_data:
            save_results(data_path, po_objs, mode="F")
            save_results(data_path, conf2, mode="Theta")
            save_results(data_path, np.array([time_cost]), mode="time")

        # add WUN
        objs, conf = weighted_utopia_nearest_impl(po_objs, conf2)
        print(f"FUNCTION: time cost of {algo} with WUN " f"is: {time.time() - start}")

        return objs, conf

    def _ws(
        self,
        len_theta_per_qs: int,
        graph_embeddings: th.Tensor,
        non_decision_df: pd.DataFrame,
        non_decision_tabular_features: th.Tensor,
        n_stages: int,
        seed: Optional[int],
        use_ag: bool,
        ag_model: Dict[str, str],
        algo: str,
        query_id: Optional[str],
        save_data: bool,
        n_samples_per_param: int,
        n_ws: int,
        time_limit: int,
        save_data_header: str,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        start = time.time()
        if use_ag:
            normalize = False
        else:
            normalize = True
        theta_s_samples = self.sample_theta_x(
            1, "s", seed + 2 if seed is not None else None, normalize=normalize
        )
        theta_s: Union[np.ndarray, th.Tensor]
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

        non_decision_features: Union[th.Tensor, pd.DataFrame]
        obj_model: Union[
            Callable[
                [np.ndarray, pd.DataFrame, np.ndarray, Dict[str, str]], np.ndarray
            ],
            Callable[[th.Tensor, th.Tensor, th.Tensor, str], np.ndarray],
        ]
        if use_ag:
            obj_model = self.get_objective_values_ag_arr
            non_decision_features = non_decision_df
        else:
            obj_model = self.get_objective_values_mlp_arr
            non_decision_features = non_decision_tabular_features

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
                conf2 = self.sc.construct_configuration(conf_qs0.astype(float)).reshape(
                    -1, len_theta_per_qs
                )

            else:
                conf2 = self.sc.construct_configuration_from_norm(conf_qs0).reshape(
                    -1, len_theta_per_qs
                )

        if use_ag:
            data_path = (
                f"{save_data_header}/latest_model_{self.device.type}/ag/{algo}/{n_samples_per_param}_{n_ws}/time_{time_limit}/"
                f"query_{query_id}_n_{n_stages}/"
            )
        else:
            data_path = (
                f"{save_data_header}/latest_model_{self.device.type}/mlp/{algo}/{n_samples_per_param}_{n_ws}/time_{time_limit}/"
                f"query_{query_id}_n_{n_stages}/"
            )

        if save_data:
            save_results(data_path, po_objs, mode="F")
            save_results(data_path, conf2, mode="Theta")
            save_results(data_path, np.array([time_cost]), mode="time")

        # add WUN
        objs, conf = weighted_utopia_nearest_impl(po_objs, conf2)
        print(f"FUNCTION: time cost of {algo} with WUN " f"is: {time.time() - start}")

        return objs, conf
