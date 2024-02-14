import time
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import torch as th
from autogluon.core import TabularDataset
from autogluon.tabular import TabularPredictor
from torchmetrics import WeightedMeanAbsolutePercentageError
from udao.model import UdaoModel, UdaoModule
from udao.model.utils.losses import WMAPELoss
from udao.optimization.utils.moo_utils import get_default_device

from udao_trace.utils import JsonHandler

from ..utils.collaborators import TypeAdvisor
from ..utils.logging import logger
from ..utils.params import QType
from .utils import (
    GraphAverageMLPParams,
    GraphTransformerMLPParams,
    calibrate_negative_predictions,
    get_graph_avg_mlp,
    get_graph_gtn_mlp,
)


class ModelServer:
    @classmethod
    def from_ckp_path(
        cls, model_sign: str, model_params_path: str, weights_path: str
    ) -> "ModelServer":
        if model_sign == "graph_avg":
            graph_avg_ml_params = GraphAverageMLPParams.from_dict(
                JsonHandler.load_json(model_params_path)
            )
            objectives = graph_avg_ml_params.iterator_shape.output_names
            model = get_graph_avg_mlp(graph_avg_ml_params)
            logger.info("GRAPH MODEL DETAILS:\n")
            logger.info(model)
        elif model_sign == "graph_gtn":
            graph_gtn_ml_params = GraphTransformerMLPParams.from_dict(
                JsonHandler.load_json(model_params_path)
            )
            objectives = graph_gtn_ml_params.iterator_shape.output_names
            model = get_graph_gtn_mlp(graph_gtn_ml_params)
            logger.info("GRAPH MODEL DETAILS:\n")
            logger.info(model)
        else:
            raise ValueError(f"Unknown model sign: {model_sign}")

        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        module = UdaoModule.load_from_checkpoint(
            weights_path,
            map_location=get_default_device(),
            model=model,
            objectives=objectives,
            loss=WMAPELoss(),
            metrics=[WeightedMeanAbsolutePercentageError],
        )
        return cls(model_sign, module)

    def __init__(self, model_sign: str, module: UdaoModule):
        self.model_sign = model_sign
        self.module = module
        if not isinstance(module.model, UdaoModel):
            raise TypeError(f"Unknown model type: {type(module.model)}")
        self.model: UdaoModel = module.model
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.objectives = module.objectives
        logger.info(f"Model loaded with objectives: {self.objectives}")


class AGServer:
    @classmethod
    def from_ckp_path(
        cls,
        model_sign: str,
        graph_model_params_path: str,
        graph_weights_path: str,
        q_type: QType,
        ag_path: str,
    ) -> "AGServer":
        ms = ModelServer.from_ckp_path(
            model_sign, graph_model_params_path, graph_weights_path
        )
        ta = TypeAdvisor(q_type=q_type)
        predictors = {
            obj: TabularPredictor.load(f"{ag_path}/Predictor_{obj}")
            for obj in ta.get_ag_objectives()
        }
        return cls(ta, ms, predictors)

    def __init__(
        self, ta: TypeAdvisor, ms: ModelServer, predictors: Dict[str, TabularPredictor]
    ):
        self.ta = ta
        self.ms = ms
        self.predictors = predictors
        self.objectives = self.ta.get_ag_objectives()
        for obj in predictors:
            self.predictors[obj].persist()  # persist the predictor in memory

    def predict_with_mlp(
        self, graph_embedding: th.Tensor, tabular_features: th.Tensor
    ) -> th.Tensor:
        """
        return multiple objective values for a given graph embedding and
        tabular features in the normalized space
        """
        with th.no_grad():
            return (
                self.ms.model.regressor(graph_embedding, tabular_features)
                .detach()
                .cpu()
            )

    def predict_with_ag(
        self,
        bm: str,
        graph_embeddings: np.ndarray,
        non_decision_df: pd.DataFrame,
        decision_variables: List[str],
        sampled_theta: np.ndarray,
        model_name: Union[str, Dict[str, str]],
    ) -> Dict[str, np.ndarray]:
        df = non_decision_df.copy()
        ge_dim = graph_embeddings.shape[1]
        ge_cols = [f"ge_{i}" for i in range(ge_dim)]
        df[ge_cols] = graph_embeddings
        df[decision_variables] = sampled_theta
        dataset = TabularDataset(df[ge_cols + self.ta.get_tabular_columns()])
        start = time.perf_counter_ns()
        transformed_data = self.predictors[self.objectives[0]].transform_features(
            dataset
        )
        dt = time.perf_counter_ns() - start
        logger.debug(f"Transformed data for AG prediction in {dt / 1e6} ms")
        return {
            obj: calibrate_negative_predictions(
                np.array(
                    self.predictors[obj].predict(
                        transformed_data,
                        model=model_name
                        if isinstance(model_name, str)
                        else model_name[obj],
                        transform_features=False,
                        as_pandas=False,
                    ),
                    dtype=np.float32,
                ),
                bm=bm,
                obj=obj,
                q_type=self.ta.q_type,
            )
            for obj in self.ta.get_ag_objectives()
        }
