import os.path
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch as th
from udao.data.handler.data_processor import DataProcessor
from udao.data.iterators.query_plan_iterator import QueryPlanInput
from udao.optimization.utils.moo_utils import Point, get_default_device

from udao_trace.configuration import SparkConf
from udao_trace.utils import PickleHandler

from ..model.model_server import AGServer
from ..utils.constants import THETA_C, THETA_COMPILE, THETA_P, THETA_S
from ..utils.logging import logger
from ..utils.params import QType
from .utils import get_cloud_cost_add_io, get_cloud_cost_wo_io

ThetaType = Literal["c", "p", "s"]


class BaseOptimizer(ABC):
    def __init__(
        self,
        bm: str,
        model_sign: str,
        graph_model_params_path: str,
        graph_weights_path: str,
        q_type: QType,
        data_processor_path: str,
        spark_conf: SparkConf,
        decision_variables: List[str],
        ag_path: str,
        verbose: bool = False,
    ) -> None:
        self.bm = bm
        self.ag_ms = AGServer.from_ckp_path(
            model_sign, graph_model_params_path, graph_weights_path, q_type, ag_path
        )
        self.ta = self.ag_ms.ta
        data_processor = PickleHandler.load(
            os.path.dirname(data_processor_path), os.path.basename(data_processor_path)
        )
        if not isinstance(data_processor, DataProcessor):
            raise TypeError(f"Expected DataHandler, got {type(data_processor)}")
        if (
            "tabular_features" not in data_processor.feature_extractors
            or "objectives" not in data_processor.feature_extractors
        ):
            raise ValueError("DataHandler must contain tabular_features and objectives")
        self.data_processor = data_processor
        feature_extractors = self.data_processor.feature_extractors
        self.tabular_columns = feature_extractors["tabular_features"].columns
        self.model_objective_columns = feature_extractors["objectives"].columns
        self.sc = spark_conf
        if decision_variables != self.tabular_columns[-len(decision_variables) :]:
            raise ValueError(
                "Decision variables must be the last columns in tabular_features"
            )
        if not all(v in THETA_COMPILE for v in decision_variables):
            raise ValueError(
                f"Decision variables must be in {THETA_COMPILE}, "
                f"got {decision_variables}"
            )
        self.decision_variables = decision_variables
        self.dtype = th.float32
        self.device = get_default_device()

        self.theta_all_minmax = (
            np.array(spark_conf.knob_min),
            np.array(spark_conf.knob_max),
        )
        self.theta_minmax = {
            "c": (
                np.array(spark_conf.knob_min[: len(THETA_C)]),
                np.array(spark_conf.knob_max[: len(THETA_C)]),
            ),
            "p": (
                np.array(
                    spark_conf.knob_min[len(THETA_C) : len(THETA_C) + len(THETA_P)]
                ),
                np.array(
                    spark_conf.knob_max[len(THETA_C) : len(THETA_C) + len(THETA_P)]
                ),
            ),
            "s": (
                np.array(spark_conf.knob_min[-len(THETA_S) :]),
                np.array(spark_conf.knob_max[-len(THETA_S) :]),
            ),
        }
        self.theta_ktype = {
            "c": [k.ktype for k in spark_conf.knob_list[: len(THETA_C)]],
            "p": [
                k.ktype
                for k in spark_conf.knob_list[
                    len(THETA_C) : len(THETA_C) + len(THETA_P)
                ]
            ],
            "s": [k.ktype for k in spark_conf.knob_list[-len(THETA_S) :]],
        }
        self.verbose = verbose

    def extract_non_decision_embeddings_from_df(
        self, df: pd.DataFrame
    ) -> Tuple[th.Tensor, th.Tensor]:
        """
        compute the graph_embedding and
        the normalized values of the non-decision variables
        """

        t1 = time.perf_counter_ns()

        if df.index.name != "id":
            raise ValueError(">>> df must have an index named 'id'")
        n_items = len(df)
        df[self.decision_variables] = 0.0
        df[self.model_objective_columns] = 0.0

        t2 = time.perf_counter_ns()
        if self.verbose:
            logger.info(f">>> preprocessed df in {(t2 - t1) / 1e6} ms")

        with th.no_grad():
            iterator = self.data_processor.make_iterator(df, df.index, split="test")

        t3 = time.perf_counter_ns()
        if self.verbose:
            logger.info(f">>> created iterator in {(t3 - t2) / 1e6} ms")

        dataloader = iterator.get_dataloader(batch_size=n_items)
        batch_input, _ = next(iter(dataloader))
        if not isinstance(batch_input, QueryPlanInput):
            raise TypeError(f"Expected QueryPlanInput, got {type(batch_input)}")

        t4 = time.perf_counter_ns()
        if self.verbose:
            logger.info(f">>> created dataloader in {(t4 - t3) / 1e6} ms")

        embedding_input = batch_input.embedding_input
        tabular_input = batch_input.features
        graph_embedding = self.ag_ms.ms.model.embedder(embedding_input.to(self.device))
        non_decision_tabular_features = tabular_input[
            :, : -len(self.decision_variables)
        ]

        t5 = time.perf_counter_ns()
        if self.verbose:
            logger.info(f">>> computed graph_embedding in {(t5 - t4) / 1e6} ms")

        return graph_embedding, non_decision_tabular_features

    def summarize_obj(
        self,
        k1: np.ndarray,
        k2: np.ndarray,
        k3: np.ndarray,
        obj_lat: np.ndarray,  # lat or ana_lat
        obj_io: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        obj_cost_wo_io = get_cloud_cost_wo_io(
            lat=obj_lat,
            cores=k1,
            mem=k1 * k2,
            nexec=k3,
        )
        obj_cost_w_io = get_cloud_cost_add_io(obj_cost_wo_io, obj_io)
        if not isinstance(obj_cost_wo_io, np.ndarray) or not isinstance(
            obj_cost_w_io, np.ndarray
        ):
            raise TypeError(
                f"Expected np.ndarray, "
                f"got {type(obj_cost_wo_io)} and {type(obj_cost_w_io)}"
            )
        if "ana_latency_s" in self.ta.get_ag_objectives():
            return {
                "ana_latency": obj_lat,
                "io": obj_io,
                "ana_cost_wo_io": obj_cost_wo_io,
                "ana_cost_w_io": obj_cost_w_io,
            }
        else:
            return {
                "latency": obj_lat,
                "io": obj_io,
                "cost_wo_io": obj_cost_wo_io,
                "cost_w_io": obj_cost_w_io,
            }

    def get_latencies_and_objectives(
        self, objs_dict: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        if "ana_latency" in objs_dict:
            return objs_dict["ana_latency"], objs_dict["ana_cost_w_io"]
        else:
            return objs_dict["latency"], objs_dict["cost_w_io"]

    def predict_objectives_mlp(
        self, graph_embedding: th.Tensor, tabular_features: th.Tensor
    ) -> th.Tensor:
        """
        return multiple objective values for a given graph embedding and
        tabular features in the normalized space
        """
        return self.ag_ms.predict_with_mlp(
            graph_embedding, tabular_features.to(self.device)
        )

    def get_objective_values_mlp(
        self,
        graph_embeddings: th.Tensor,
        non_decision_tabular_features: th.Tensor,
        theta: th.Tensor,
    ) -> Dict[str, np.ndarray]:
        tabular_features = th.cat([non_decision_tabular_features, theta], dim=1)
        objs = self.predict_objectives_mlp(graph_embeddings, tabular_features).numpy()
        theta_c_min, theta_c_max = self.theta_minmax["c"]
        k1_min, k2_min, k3_min = theta_c_min[:3]
        k1_max, k2_max, k3_max = theta_c_max[:3]

        k1_norm = tabular_features[:, -len(THETA_C + THETA_P + THETA_S)].numpy()
        k2_norm = tabular_features[:, -len(THETA_C + THETA_P + THETA_S) + 1].numpy()
        k3_norm = tabular_features[:, -len(THETA_C + THETA_P + THETA_S) + 2].numpy()
        k1 = (k1_norm - k1_min) * (k1_max - k1_min) + k1_min
        k2 = (k2_norm - k2_min) * (k2_max - k2_min) + k2_min
        k3 = (k3_norm - k3_min) * (k3_max - k3_min) + k3_min

        if self.ta.q_type.startswith("qs_"):
            obj_io = objs[:, 1]
            obj_ana_lat = objs[:, 2]
            return self.summarize_obj(k1, k2, k3, obj_ana_lat, obj_io)
        else:
            obj_lat = objs[:, 0]
            obj_io = objs[:, 1]
            return self.summarize_obj(k1, k2, k3, obj_lat, obj_io)

    def get_objective_values_ag(
        self,
        graph_embeddings: np.ndarray,
        non_decision_df: pd.DataFrame,
        sampled_theta: np.ndarray,
        model_name: Union[str, Dict[str, str]],
    ) -> Dict[str, np.ndarray]:
        start_time_ns = time.perf_counter_ns()
        objs = self.ag_ms.predict_with_ag(
            self.bm,
            graph_embeddings,
            non_decision_df,
            self.decision_variables,
            sampled_theta,
            model_name,
        )
        end_time_ns = time.perf_counter_ns()
        logger.info(
            f"Takes {(end_time_ns - start_time_ns) / 1e6} ms "
            f"to compute {len(sampled_theta)} theta"
        )

        if "k1" in non_decision_df.columns:
            k1 = non_decision_df["k1"].values
            k2 = non_decision_df["k2"].values
            k3 = non_decision_df["k3"].values
        elif "k1" in self.decision_variables:
            if sampled_theta.ndim == 1:
                raise ValueError("sampled_theta must be 2D")
            k1 = sampled_theta[:, 0]
            k2 = sampled_theta[:, 1]
            k3 = sampled_theta[:, 2]
        else:
            raise ValueError("k1, k2, k3 not found")

        obj_lat = (
            objs["ana_latency_s"]
            if self.ta.q_type.startswith("qs_")
            else objs["latency_s"]
        )
        obj_io = objs["io_mb"]
        return self.summarize_obj(
            np.array(k1), np.array(k2), np.array(k3), obj_lat, obj_io
        )

    def sample_theta_all(
        self, n_samples: int, seed: Optional[int], normalize: bool
    ) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)
        samples = np.random.randint(
            low=self.theta_all_minmax[0],
            high=self.theta_all_minmax[1],
            size=(n_samples, len(self.sc.knob_min)),
        )
        if normalize:
            samples_normalized = (samples - self.theta_all_minmax[0]) / (
                self.theta_all_minmax[1] - self.theta_all_minmax[0]
            )
            return samples_normalized
        else:
            return samples

    def sample_theta_x(
        self,
        n_samples: int,
        theta_type: ThetaType,
        seed: Optional[int],
        normalize: bool = True,
    ) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)
        samples = np.random.randint(
            low=self.theta_minmax[theta_type][0],
            high=self.theta_minmax[theta_type][1],
            size=(n_samples, len(self.theta_minmax[theta_type][0])),
        )
        if normalize:
            samples_normalized = (samples - self.theta_minmax[theta_type][0]) / (
                self.theta_minmax[theta_type][1] - self.theta_minmax[theta_type][0]
            )
            return samples_normalized
        else:
            return samples

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

    @abstractmethod
    def solve(
        self,
        non_decision_input: Dict[str, Any],
        seed: Optional[int] = None,
        use_ag: bool = True,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        pass

    def weighted_utopia_nearest(self, pareto_points: List[Point]) -> Point:
        """
        return the Pareto point that is closest to the utopia point
        in a weighted distance function
        """
        # todo
        pass
