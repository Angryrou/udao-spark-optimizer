import os.path
from abc import ABC, abstractmethod
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import torch as th
from udao.data.handler.data_processor import DataProcessor
from udao.data.iterators.query_plan_iterator import QueryPlanInput
from udao.optimization.utils.moo_utils import Point, get_default_device

from udao_trace.configuration import SparkConf
from udao_trace.utils import PickleHandler

from ..model.model_server import ModelServer
from ..utils.constants import THETA_C, THETA_P, THETA_S

ThetaType = Literal["c", "p", "s"]


class BaseOptimizer(ABC):
    def __init__(
        self,
        model_sign: str,
        model_params_path: str,
        weights_path: str,
        data_processor_path: str,
        spark_conf: SparkConf,
        decision_variables: List[str],
    ) -> None:
        self.ms = ModelServer.from_ckp_path(model_sign, model_params_path, weights_path)
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

    def extract_non_decision_embeddings_from_df(
        self, df: pd.DataFrame
    ) -> Tuple[th.Tensor, th.Tensor]:
        """
        compute the graph_embedding and
        the normalized values of the non-decision variables
        """
        if df.index.name != "id":
            raise ValueError("df must have an index named 'id'")
        n_items = len(df)
        df[self.decision_variables] = 0.0
        df[self.model_objective_columns] = 0.0
        with th.no_grad():
            iterator = self.data_processor.make_iterator(df, df.index, split="test")
        dataloader = iterator.get_dataloader(batch_size=n_items)
        batch_input, _ = next(iter(dataloader))
        if not isinstance(batch_input, QueryPlanInput):
            raise TypeError(f"Expected QueryPlanInput, got {type(batch_input)}")
        embedding_input = batch_input.embedding_input
        tabular_input = batch_input.features
        graph_embedding = self.ms.model.embedder(embedding_input.to(self.device))
        non_decision_tabular_features = tabular_input[
            :, : -len(self.decision_variables)
        ]
        return graph_embedding, non_decision_tabular_features

    def _predict_objectives(
        self, graph_embedding: th.Tensor, tabular_features: th.Tensor
    ) -> th.Tensor:
        """
        return multiple objective values for a given graph embedding and
        tabular features in the normalized space
        """
        with th.no_grad():
            return (
                self.ms.model.regressor(
                    graph_embedding, tabular_features.to(self.device)
                )
                .detach()
                .cpu()
            )

    def sample_theta_all(self, n_samples: int, seed: Optional[int]) -> th.Tensor:
        if seed is not None:
            np.random.seed(seed)
        samples = np.random.randint(
            low=self.theta_all_minmax[0],
            high=self.theta_all_minmax[1] + 1,
            size=(n_samples, len(self.sc.knob_min)),
        )
        samples_normalized = (samples - self.theta_all_minmax[0]) / (
            self.theta_all_minmax[1] - self.theta_all_minmax[0]
        )
        return th.tensor(samples_normalized, dtype=self.dtype)

    def sample_theta_x(
        self, n_samples: int, theta_type: ThetaType, seed: Optional[int]
    ) -> th.Tensor:
        if seed is not None:
            np.random.seed(seed)
        samples = np.random.randint(
            low=self.theta_minmax[theta_type][0],
            high=self.theta_minmax[theta_type][1],
            size=(n_samples, len(self.theta_minmax[theta_type][0])),
        )
        samples_normalized = (samples - self.theta_minmax[theta_type][0]) / (
            self.theta_minmax[theta_type][1] - self.theta_minmax[theta_type][0]
        )
        return th.tensor(samples_normalized, dtype=self.dtype)

    @abstractmethod
    def solve(
        self, non_decision_input: Dict, seed: Optional[int] = None
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        pass

    def weighted_utopia_nearest(self, pareto_points: List[Point]) -> Point:
        """
        return the Pareto point that is closest to the utopia point
        in a weighted distance function
        """
        # todo
        pass
