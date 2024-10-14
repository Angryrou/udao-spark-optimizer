import glob
import hashlib
import os.path
import time
from argparse import Namespace
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import dgl
import lightning.pytorch as pl
import networkx as nx
import numpy as np
import pandas as pd
import pytorch_warmup as warmup
import torch as th
import udao
from autogluon.core.metrics import make_scorer
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torchmetrics import Metric, WeightedMeanAbsolutePercentageError
from udao.data import BaseIterator, QueryPlanIterator
from udao.data.iterators.query_plan_iterator import QueryPlanInput
from udao.data.utils.query_plan import QueryPlanStructure
from udao.data.utils.utils import DatasetType
from udao.model import MLP, UdaoModel
from udao.model.embedders.layers.multi_head_attention import AttentionLayerName
from udao.model.module import LearningParams
from udao.model.utils.losses import WMAPELoss
from udao.model.utils.schedulers import UdaoLRScheduler, setup_cosine_annealing_lr
from udao.optimization.utils.moo_utils import get_default_device
from udao.utils.interfaces import UdaoEmbedItemShape

from udao_trace.utils import JsonHandler, PickleHandler

from ..data.utils import checkpoint_model_structure
from ..utils.collaborators import PathWatcher, TypeAdvisor
from ..utils.logging import logger
from ..utils.params import ExtractParams, QType, UdaoParams
from .embedders import GraphAverager, GraphTransformer, QPPNet, TreeCNN, TreeLSTM
from .regressors.basic_mlp import BasicMLP
from .regressors.qppnet_out import QPPNetOut
from .regressors.sk_mlp import SkipConnectionMLP


class UdaoModule(udao.model.UdaoModule):
    def training_step(self, batch: Tuple[Any, th.Tensor], batch_idx: int) -> th.Tensor:
        y_hat, y = self._shared_step(batch, "train")
        loss, _ = self.compute_loss(y, y_hat)
        if th.isnan(loss):
            raise ValueError("got a nan loss in train")
        elif th.isinf(loss):
            raise ValueError("got an inf loss in train")
        self.log(
            "learning_rate",
            self.trainer.optimizers[0].param_groups[0]["lr"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def on_validation_epoch_end(self) -> None:
        val_loss = 0.0
        for objective in self.objectives:
            metric = cast(Metric, self.metrics[objective])
            output = metric.compute()
            for k, v in output.items():
                if k == f"{objective}_WeightedMeanAbsolutePercentageError":
                    val_loss += self.loss_weights[objective] * float(v)

        self.log(
            "val_loss",
            val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self._shared_epoch_end("val")


@dataclass
class MyLearningParams(UdaoParams):
    epochs: int = 2
    batch_size: int = 512
    init_lr: float = 1e-1
    min_lr: float = 1e-5
    weight_decay: float = 1e-2
    loss_weights: Optional[List[float]] = None

    @classmethod
    def from_dict(cls, data_dict: Dict[str, Any]) -> "MyLearningParams":
        return cls(**data_dict)

    def hash(self) -> str:
        attributes_tuple = ",".join(
            f"{x:g}" if isinstance(x, float) else str(x)
            for x in (
                self.epochs,
                self.batch_size,
                self.init_lr,
                self.min_lr,
                self.weight_decay,
            )
        ).encode("utf-8")
        sha256_hash = hashlib.sha256(attributes_tuple)
        hex12 = sha256_hash.hexdigest()[:12]
        if self.loss_weights is not None:
            loss_weights_str = "_".join(f"{v:g}" for v in self.loss_weights)
            return f"learning_{hex12}_{loss_weights_str}"
        else:
            return "learning_" + hex12


@dataclass
class GraphAverageMLPParams(UdaoParams):
    iterator_shape: UdaoEmbedItemShape
    op_groups: List[str]
    output_size: int = 32
    type_embedding_dim: int = 8
    hist_embedding_dim: Optional[int] = None
    bitmap_embedding_dim: Optional[int] = None
    embedding_normalizer: Optional[str] = None
    # MLP
    n_layers: int = 2
    hidden_dim: int = 32
    dropout: float = 0.1

    @classmethod
    def from_dict(cls, data_dict: Dict[str, Any]) -> "GraphAverageMLPParams":
        if "iterator_shape" not in data_dict:
            raise ValueError("iterator_shape not found in data_dict")
        if not isinstance(data_dict["iterator_shape"], UdaoEmbedItemShape):
            iterator_shape_dict = data_dict["iterator_shape"]
            data_dict["iterator_shape"] = UdaoEmbedItemShape(
                embedding_input_shape=iterator_shape_dict["embedding_input_shape"],
                feature_names=iterator_shape_dict["feature_names"],
                output_names=iterator_shape_dict["output_names"],
            )
        return cls(**data_dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            k: v if not isinstance(v, UdaoEmbedItemShape) else v.__dict__
            for k, v in self.__dict__.items()
        }

    def hash(self) -> str:
        elements = [
            str(self.iterator_shape),
            tuple(self.op_groups),
            self.output_size,
            self.type_embedding_dim,
        ]

        # Conditionally add elements if they are not None
        if self.hist_embedding_dim is not None:
            elements.append(self.hist_embedding_dim)

        if self.bitmap_embedding_dim is not None:
            elements.append(self.bitmap_embedding_dim)

        # Continue adding the rest of the elements
        elements.extend(
            [self.embedding_normalizer, self.n_layers, self.hidden_dim, self.dropout]
        )

        # Convert the list to a tuple, then to string, and encode it
        attributes_tuple = str(tuple(elements)).encode("utf-8")
        sha256_hash = hashlib.sha256(attributes_tuple)
        hex12 = sha256_hash.hexdigest()[:12]
        return "graph_avg_" + hex12


def get_graph_avg_mlp(params: GraphAverageMLPParams) -> UdaoModel:
    model = UdaoModel.from_config(
        embedder_cls=GraphAverager,
        regressor_cls=MLP,
        iterator_shape=params.iterator_shape,
        embedder_params={
            "output_size": params.output_size,  # 32
            "op_groups": params.op_groups,  # ["type", "cbo", "op_enc"]
            "type_embedding_dim": params.type_embedding_dim,  # 8
            "hist_embedding_dim": params.hist_embedding_dim,  # 32
            "bitmap_embedding_dim": params.bitmap_embedding_dim,  # 32
            "embedding_normalizer": params.embedding_normalizer,  # None
        },
        regressor_params={
            "n_layers": params.n_layers,  # 3
            "hidden_dim": params.hidden_dim,  # 512
            "dropout": params.dropout,  # 0.1
        },
    )
    return model


@dataclass
class GraphTransformerMLPParams(UdaoParams):
    iterator_shape: UdaoEmbedItemShape
    op_groups: List[str]
    output_size: int = 32
    pos_encoding_dim: int = 8
    gtn_n_layers: int = 2
    gtn_n_heads: int = 2
    readout: str = "mean"
    type_embedding_dim: int = 8
    hist_embedding_dim: int = 32
    bitmap_embedding_dim: int = 32
    embedding_normalizer: Optional[str] = None
    attention_layer_name: AttentionLayerName = "GTN"
    # For QF (QueryFormer)
    max_dist: Optional[int] = None
    max_height: Optional[int] = None
    # For RAAL
    non_siblings_map: Optional[Dict[int, Dict[int, List[int]]]] = None
    # MLP
    n_layers: int = 2
    hidden_dim: int = 32
    dropout: float = 0.1

    @classmethod
    def from_dict(cls, data_dict: Dict[str, Any]) -> "GraphTransformerMLPParams":
        if "iterator_shape" not in data_dict:
            raise ValueError("iterator_shape not found in data_dict")
        if not isinstance(data_dict["iterator_shape"], UdaoEmbedItemShape):
            iterator_shape_dict = data_dict["iterator_shape"]
            data_dict["iterator_shape"] = UdaoEmbedItemShape(
                embedding_input_shape={
                    k: v
                    for k, v in iterator_shape_dict["embedding_input_shape"].items()
                    if k in data_dict["op_groups"]
                },
                feature_names=iterator_shape_dict["feature_names"],
                output_names=iterator_shape_dict["output_names"],
            )
        else:
            data_dict["iterator_shape"].embedding_input_shape = {
                k: v
                for k, v in data_dict["iterator_shape"].embedding_input_shape.items()
                if k in data_dict["op_groups"]
            }
        if "attention_layer_name" in data_dict:
            if (
                data_dict["attention_layer_name"] == "RAAL"
                and "non_siblings_map" not in data_dict
            ):
                raise ValueError("non_siblings_map not found for RAAL")
            if (
                data_dict["attention_layer_name"] == "QF"
                and "max_dist" not in data_dict
            ):
                raise ValueError("max_dist not found for QF")
            if (
                data_dict["attention_layer_name"] == "QF"
                and "max_height" not in data_dict
            ):
                raise ValueError("max_height not found for QF")
        return cls(**data_dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            k: v if not isinstance(v, UdaoEmbedItemShape) else v.__dict__
            for k, v in self.__dict__.items()
            if v is not None and k not in ["non_siblings_map"]
        }

    def hash(self) -> str:
        elements = [
            str(self.iterator_shape),
            tuple(self.op_groups),
            self.output_size,
            self.pos_encoding_dim,
            self.gtn_n_layers,
            self.gtn_n_heads,
            self.readout,
            self.type_embedding_dim,
        ]

        # Conditionally add elements if they are not None
        if self.hist_embedding_dim is not None:
            elements.append(self.hist_embedding_dim)

        if self.bitmap_embedding_dim is not None:
            elements.append(self.bitmap_embedding_dim)

        # Continue adding the rest of the elements
        elements.extend(
            [self.embedding_normalizer, self.n_layers, self.hidden_dim, self.dropout]
        )

        # Convert the list to a tuple, then to string, and encode it
        attributes_tuple = str(tuple(elements)).encode("utf-8")
        sha256_hash = hashlib.sha256(attributes_tuple)
        hex12 = sha256_hash.hexdigest()[:12]
        return f"graph_{self.attention_layer_name.lower()}_" + hex12


def get_graph_transformer_mlp(params: GraphTransformerMLPParams) -> UdaoModel:
    model = UdaoModel.from_config(
        embedder_cls=GraphTransformer,
        regressor_cls=MLP,
        iterator_shape=params.iterator_shape,
        embedder_params={
            "output_size": params.output_size,  # 128
            "pos_encoding_dim": params.pos_encoding_dim,  # 8
            "n_layers": params.gtn_n_layers,  # 2
            "n_heads": params.gtn_n_heads,  # 2
            "hidden_dim": params.output_size,  # same as out_size
            "readout": params.readout,  # "mean"
            "op_groups": params.op_groups,  # ["type", "cbo", "op_enc"]
            "type_embedding_dim": params.type_embedding_dim,  # 8
            "hist_embedding_dim": params.hist_embedding_dim,  # 32
            "bitmap_embedding_dim": params.bitmap_embedding_dim,  # 32
            "embedding_normalizer": params.embedding_normalizer,  # None
            "attention_layer_name": params.attention_layer_name,  # "GTN"
            "max_dist": params.max_dist,  # None
            "max_height": params.max_height,  # None
            "non_siblings_map": params.non_siblings_map,  # None
        },
        regressor_params={
            "n_layers": params.n_layers,  # 3
            "hidden_dim": params.hidden_dim,  # 512
            "dropout": params.dropout,  # 0.1
        },
    )
    return model


@dataclass
class GraphTransformerBasicMLPParams(UdaoParams):
    iterator_shape: UdaoEmbedItemShape
    op_groups: List[str]
    output_size: int = 32
    pos_encoding_dim: int = 8
    gtn_n_layers: int = 2
    gtn_n_heads: int = 2
    readout: str = "mean"
    type_embedding_dim: int = 8
    hist_embedding_dim: int = 32
    bitmap_embedding_dim: int = 32
    embedding_normalizer: Optional[str] = None
    attention_layer_name: AttentionLayerName = "GTN"
    gtn_dropout: float = 0.0
    # For QF (QueryFormer)
    max_dist: Optional[int] = None
    max_height: Optional[int] = None
    # For RAAL
    non_siblings_map: Optional[Dict[int, Dict[int, List[int]]]] = None
    # MLP
    n_layers: int = 2
    hidden_dim: int = 32
    dropout: float = 0.1
    use_batchnorm: bool = True
    activate: str = "relu"

    @classmethod
    def from_dict(cls, data_dict: Dict[str, Any]) -> "GraphTransformerBasicMLPParams":
        if "iterator_shape" not in data_dict:
            raise ValueError("iterator_shape not found in data_dict")
        if not isinstance(data_dict["iterator_shape"], UdaoEmbedItemShape):
            iterator_shape_dict = data_dict["iterator_shape"]
            data_dict["iterator_shape"] = UdaoEmbedItemShape(
                embedding_input_shape={
                    k: v
                    for k, v in iterator_shape_dict["embedding_input_shape"].items()
                    if k in data_dict["op_groups"]
                },
                feature_names=iterator_shape_dict["feature_names"],
                output_names=iterator_shape_dict["output_names"],
            )
        else:
            data_dict["iterator_shape"].embedding_input_shape = {
                k: v
                for k, v in data_dict["iterator_shape"].embedding_input_shape.items()
                if k in data_dict["op_groups"]
            }
        if "attention_layer_name" in data_dict:
            if (
                data_dict["attention_layer_name"] == "RAAL"
                and "non_siblings_map" not in data_dict
            ):
                raise ValueError("non_siblings_map not found for RAAL")
            if (
                data_dict["attention_layer_name"] == "QF"
                and "max_dist" not in data_dict
            ):
                raise ValueError("max_dist not found for QF")
            if (
                data_dict["attention_layer_name"] == "QF"
                and "max_height" not in data_dict
            ):
                raise ValueError("max_height not found for QF")
        return cls(**data_dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            k: v if not isinstance(v, UdaoEmbedItemShape) else v.__dict__
            for k, v in self.__dict__.items()
            if v is not None and k not in ["non_siblings_map"]
        }

    def hash(self) -> str:
        attributes_tuple = str(
            (
                str(self.iterator_shape),
                tuple(self.op_groups),
                self.output_size,
                self.pos_encoding_dim,
                self.gtn_n_layers,
                self.gtn_n_heads,
                self.readout,
                self.type_embedding_dim,
                self.hist_embedding_dim,
                self.bitmap_embedding_dim,
                self.embedding_normalizer,
                self.gtn_dropout,
                self.n_layers,
                self.hidden_dim,
                self.dropout,
                self.use_batchnorm,
                self.activate,
            )
        ).encode("utf-8")
        sha256_hash = hashlib.sha256(attributes_tuple)
        hex12 = sha256_hash.hexdigest()[:12]
        return f"graph_{self.attention_layer_name.lower()}_basic_mlp" + hex12


def get_graph_transformer_basic_mlp(
    params: GraphTransformerBasicMLPParams,
) -> UdaoModel:
    model = UdaoModel.from_config(
        embedder_cls=GraphTransformer,
        regressor_cls=BasicMLP,
        iterator_shape=params.iterator_shape,
        embedder_params={
            "output_size": params.output_size,  # 128
            "pos_encoding_dim": params.pos_encoding_dim,  # 8
            "n_layers": params.gtn_n_layers,  # 2
            "n_heads": params.gtn_n_heads,  # 2
            "hidden_dim": params.output_size,  # same as out_size
            "readout": params.readout,  # "mean"
            "op_groups": params.op_groups,  # all types
            "type_embedding_dim": params.type_embedding_dim,  # 8
            "hist_embedding_dim": params.hist_embedding_dim,  # 32
            "bitmap_embedding_dim": params.bitmap_embedding_dim,  # 32
            "embedding_normalizer": params.embedding_normalizer,  # None
            "attention_layer_name": params.attention_layer_name,  # "GTN"
            "dropout": params.gtn_dropout,
            "max_dist": params.max_dist,  # None
            "max_height": params.max_height,  # None
            "non_siblings_map": params.non_siblings_map,  # None
        },
        regressor_params={
            "n_layers": params.n_layers,  # 3
            "hidden_dim": params.hidden_dim,  # 512
            "dropout": params.dropout,  # 0.1
            "use_batchnorm": params.use_batchnorm,  # True
            "activation": params.activate,  # "relu"
        },
    )
    return model


@dataclass
class GraphTransformerSKMLPParams(UdaoParams):
    iterator_shape: UdaoEmbedItemShape
    op_groups: List[str]
    output_size: int = 32
    pos_encoding_dim: int = 8
    gtn_n_layers: int = 2
    gtn_n_heads: int = 2
    readout: str = "mean"
    type_embedding_dim: int = 8
    hist_embedding_dim: int = 32
    bitmap_embedding_dim: int = 32
    embedding_normalizer: Optional[str] = None
    attention_layer_name: AttentionLayerName = "GTN"
    gtn_dropout: float = 0.0
    # For QF (QueryFormer)
    max_dist: Optional[int] = None
    max_height: Optional[int] = None
    # For RAAL
    non_siblings_map: Optional[Dict[int, Dict[int, List[int]]]] = None
    # MLP
    n_layers: int = 2
    hidden_dim: int = 32
    dropout: float = 0.1
    use_batchnorm: bool = True
    activate: str = "relu"

    @classmethod
    def from_dict(cls, data_dict: Dict[str, Any]) -> "GraphTransformerSKMLPParams":
        if "iterator_shape" not in data_dict:
            raise ValueError("iterator_shape not found in data_dict")
        if not isinstance(data_dict["iterator_shape"], UdaoEmbedItemShape):
            iterator_shape_dict = data_dict["iterator_shape"]
            data_dict["iterator_shape"] = UdaoEmbedItemShape(
                embedding_input_shape={
                    k: v
                    for k, v in iterator_shape_dict["embedding_input_shape"].items()
                    if k in data_dict["op_groups"]
                },
                feature_names=iterator_shape_dict["feature_names"],
                output_names=iterator_shape_dict["output_names"],
            )
        else:
            data_dict["iterator_shape"].embedding_input_shape = {
                k: v
                for k, v in data_dict["iterator_shape"].embedding_input_shape.items()
                if k in data_dict["op_groups"]
            }
        if "attention_layer_name" in data_dict:
            if (
                data_dict["attention_layer_name"] == "RAAL"
                and "non_siblings_map" not in data_dict
            ):
                raise ValueError("non_siblings_map not found for RAAL")
            if (
                data_dict["attention_layer_name"] == "QF"
                and "max_dist" not in data_dict
            ):
                raise ValueError("max_dist not found for QF")
            if (
                data_dict["attention_layer_name"] == "QF"
                and "max_height" not in data_dict
            ):
                raise ValueError("max_height not found for QF")
        return cls(**data_dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            k: v if not isinstance(v, UdaoEmbedItemShape) else v.__dict__
            for k, v in self.__dict__.items()
            if v is not None and k not in ["non_siblings_map"]
        }

    def hash(self) -> str:
        attributes_tuple = str(
            (
                str(self.iterator_shape),
                tuple(self.op_groups),
                self.output_size,
                self.pos_encoding_dim,
                self.gtn_n_layers,
                self.gtn_n_heads,
                self.readout,
                self.type_embedding_dim,
                self.hist_embedding_dim,
                self.bitmap_embedding_dim,
                self.embedding_normalizer,
                self.gtn_dropout,
                self.n_layers,
                self.hidden_dim,
                self.dropout,
                self.use_batchnorm,
                self.activate,
            )
        ).encode("utf-8")
        sha256_hash = hashlib.sha256(attributes_tuple)
        hex12 = sha256_hash.hexdigest()[:12]
        return f"graph_{self.attention_layer_name.lower()}_sk_mlp" + hex12


def get_graph_transformer_sk_mlp(params: GraphTransformerSKMLPParams) -> UdaoModel:
    model = UdaoModel.from_config(
        embedder_cls=GraphTransformer,
        regressor_cls=SkipConnectionMLP,
        iterator_shape=params.iterator_shape,
        embedder_params={
            "output_size": params.output_size,  # 128
            "pos_encoding_dim": params.pos_encoding_dim,  # 8
            "n_layers": params.gtn_n_layers,  # 2
            "n_heads": params.gtn_n_heads,  # 2
            "hidden_dim": params.output_size,  # same as out_size
            "readout": params.readout,  # "mean"
            "op_groups": params.op_groups,  # all types
            "type_embedding_dim": params.type_embedding_dim,  # 8
            "hist_embedding_dim": params.hist_embedding_dim,  # 32
            "bitmap_embedding_dim": params.bitmap_embedding_dim,  # 32
            "embedding_normalizer": params.embedding_normalizer,  # None
            "attention_layer_name": params.attention_layer_name,  # "GTN"
            "dropout": params.gtn_dropout,
            "max_dist": params.max_dist,  # None
            "max_height": params.max_height,  # None
            "non_siblings_map": params.non_siblings_map,  # None
        },
        regressor_params={
            "n_layers": params.n_layers,  # 3
            "hidden_dim": params.hidden_dim,  # 512
            "dropout": params.dropout,  # 0.1
            "use_batchnorm": params.use_batchnorm,  # True
            "activation": params.activate,  # "relu"
        },
    )
    return model


def sink_job_stats(
    ckp_learning_header: str, device: str, target_mask: Optional[np.ndarray] = None
) -> None:
    if not os.path.exists(
        f"{ckp_learning_header}/obj_df_test_SYNTHETIC_{device}.pkl"
    ) or not os.path.exists(f"{ckp_learning_header}/obj_df_test_LIGHT_{device}.pkl"):
        tfile = glob.glob(f"{ckp_learning_header}/obj_df_test_with_*.pkl")[0]
        tfile_name = os.path.basename(tfile)
        cache = PickleHandler.load(ckp_learning_header, tfile_name)
        if not isinstance(cache, dict):
            raise TypeError(f"Expected dict, got {type(cache)}")
        obj_df = cache["obj_df"]
        obj_names = obj_df.columns.to_list()[:2]
        if target_mask is not None:
            obj_df = obj_df[target_mask]
        ood_mask = np.array([False] * len(obj_df))
        ood_mask[-70:] = True
        id_mask = ~ood_mask
        for masks, test_type in [(id_mask, "SYNTHETIC"), (ood_mask, "LIGHT")]:
            obj_df_sub = obj_df[masks].copy()
            metrics: Dict[str, Dict[str, float]] = {}
            for obj_ind, obj in enumerate(obj_names):
                metrics[obj] = {}
                y = obj_df_sub[obj].values
                y_pred = obj_df_sub[f"{obj}_pred"].values
                qerr = np.maximum(y, y_pred) / np.minimum(y, y_pred)
                metrics[obj]["wmape"] = local_wmape(y, y_pred)
                metrics[obj]["p50_err"] = local_p50_err(y, y_pred)
                metrics[obj]["p90_err"] = local_p90_err(y, y_pred)
                metrics[obj]["p50_wape"] = local_p50_wape(y, y_pred)
                metrics[obj]["p90_wape"] = local_p90_wape(y, y_pred)
                metrics[obj]["p50_qerr"] = np.percentile(qerr, 50)
                metrics[obj]["p90_qerr"] = np.percentile(qerr, 90)
                metrics[obj]["p99_qerr"] = np.percentile(qerr, 99)
                metrics[obj]["max_qerr"] = np.max(qerr)
                metrics[obj]["mean_qerr"] = np.mean(qerr)
                metrics[obj]["corr"] = float(np.corrcoef(y, y_pred)[0, 1])
            print("metrics: ", metrics)
            test_file_name_sub = f"obj_df_test_{test_type}_{device}.pkl"
            PickleHandler.save(
                {
                    "obj_df": obj_df_sub,
                    "metrics": metrics,
                },
                ckp_learning_header,
                test_file_name_sub,
                overwrite=True,
            )


def sink_tpc_stats(
    benchmark: str,
    ckp_learning_header: str,
    device: str,
    target_mask: np.ndarray,
) -> None:
    if not os.path.exists(
        f"{ckp_learning_header}/obj_df_test_{benchmark}_{device}.pkl"
    ):
        tfile = glob.glob(f"{ckp_learning_header}/obj_df_test_with_*.pkl")[0]
        tfile_name = os.path.basename(tfile)
        cache = PickleHandler.load(ckp_learning_header, tfile_name)
        if not isinstance(cache, dict):
            raise TypeError(f"Expected dict, got {type(cache)}")
        obj_df = cache["obj_df"][target_mask]
        obj_names = obj_df.columns.to_list()[:2]

        metrics: Dict[str, Dict[str, float]] = {}
        for obj_ind, obj in enumerate(obj_names):
            metrics[obj] = {}
            y = obj_df[obj].values
            y_pred = obj_df[f"{obj}_pred"].values
            qerr = np.maximum(y, y_pred) / np.minimum(y, y_pred)
            metrics[obj]["wmape"] = local_wmape(y, y_pred)
            metrics[obj]["p50_err"] = local_p50_err(y, y_pred)
            metrics[obj]["p90_err"] = local_p90_err(y, y_pred)
            metrics[obj]["p50_wape"] = local_p50_wape(y, y_pred)
            metrics[obj]["p90_wape"] = local_p90_wape(y, y_pred)
            metrics[obj]["p50_qerr"] = np.percentile(qerr, 50)
            metrics[obj]["p90_qerr"] = np.percentile(qerr, 90)
            metrics[obj]["p99_qerr"] = np.percentile(qerr, 99)
            metrics[obj]["max_qerr"] = np.max(qerr)
            metrics[obj]["mean_qerr"] = np.mean(qerr)
            metrics[obj]["corr"] = float(np.corrcoef(y, y_pred)[0, 1])
        print("metrics: ", metrics)
        test_file_name_sub = f"obj_df_test_{benchmark}_{device}.pkl"
        PickleHandler.save(
            {"obj_df": obj_df, "metrics": metrics},
            ckp_learning_header,
            test_file_name_sub,
            overwrite=True,
        )


def param_init(params: Namespace) -> Tuple[TypeAdvisor, PathWatcher]:
    ta = TypeAdvisor(q_type=params.q_type)
    extract_params = ExtractParams.from_dict(
        {
            "lpe_size": params.lpe_size,
            "vec_size": params.vec_size,
            "seed": params.seed,
            "q_type": params.q_type,
            "debug": params.debug,
        }
    )
    pw = PathWatcher(
        Path(__file__).parent,
        params.benchmark,
        params.debug,
        extract_params,
        params.fold,
        params.data_percentage,
    )
    return ta, pw


def train_and_dump(
    ta: TypeAdvisor,
    pw: PathWatcher,
    model: UdaoModel,
    split_iterators: Dict[DatasetType, BaseIterator],
    extract_params: ExtractParams,
    model_params: UdaoParams,
    learning_params: MyLearningParams,
    params: Namespace,
    device: str,
) -> None:
    # prepare the model structure path
    tabular_columns = ta.get_tabular_columns()
    objectives = ta.get_objectives()
    logger.info(f"Tabular columns: {tabular_columns}")
    logger.info(f"Objectives: {objectives}")

    ckp_header = checkpoint_model_structure(pw=pw, model_params=model_params)
    start_time = time.perf_counter_ns()
    trainer, module, ckp_learning_header, found = get_tuned_trainer(
        ckp_header,
        model,
        split_iterators,
        objectives,
        learning_params,
        device,
        num_workers=0 if params.debug else params.num_workers,
    )

    if not found:
        train_time_secs = (time.perf_counter_ns() - start_time) / 1e9
        test_results = trainer.test(
            model=module,
            dataloaders=split_iterators["test"].get_dataloader(
                batch_size=learning_params.batch_size,
                num_workers=0 if params.debug else params.num_workers,
                shuffle=False,
            ),
        )
        JsonHandler.dump_to_file(
            {
                "test_results": test_results,
                "extract_params": extract_params.__dict__,
                "model_params": model_params.to_dict(),
                "learning_params": learning_params.__dict__,
                "tabular_columns": tabular_columns,
                "objectives": objectives,
                "training_time_s": train_time_secs,
            },
            f"{ckp_learning_header}/test_results.json",
            indent=2,
        )
        print(test_results)
        save_mlp_training_results(
            module=module,
            split_iterators=split_iterators,
            params=params,
            ckp_learning_header=ckp_learning_header,
            test_results=test_results[0],
            device=device,
        )
    else:
        print("model found and loadable at", ckp_learning_header)

    if params.benchmark == "job":
        sink_job_stats(ckp_learning_header, device)
    elif params.benchmark.endswith("+job"):
        bm_tpc = params.benchmark.split("+")[0]
        pw_job = PathWatcher(
            pw.base_dir,
            "job",
            pw.debug,
            pw.extract_params,
            None,
        )
        index_splits_jobs = PickleHandler.load(
            pw_job.cc_prefix, "index_splits_q_compile.pkl"
        )
        if not isinstance(index_splits_jobs, dict):
            raise TypeError(f"Expected dict, got {type(index_splits_jobs)}")
        n_test_jobs = len(index_splits_jobs["test"])
        n_test = len(split_iterators["test"].keys)
        job_mask = np.array([False] * n_test)
        job_mask[-n_test_jobs:] = True
        sink_job_stats(ckp_learning_header, device, job_mask)
        tpc_mask = ~job_mask
        sink_tpc_stats(bm_tpc, ckp_learning_header, device, tpc_mask)


@dataclass
class TreeLSTMParams(UdaoParams):
    iterator_shape: UdaoEmbedItemShape
    op_groups: List[str]
    output_size: int = 32
    lstm_hidden_dim: int = 32
    lstm_dropout: float = 0.0
    readout: str = "mean"
    type_embedding_dim: int = 8
    hist_embedding_dim: int = 32
    bitmap_embedding_dim: int = 32
    embedding_normalizer: Optional[str] = None
    # MLP
    n_layers: int = 2
    hidden_dim: int = 32
    dropout: float = 0.1

    @classmethod
    def from_dict(cls, data_dict: Dict[str, Any]) -> "TreeLSTMParams":
        if "iterator_shape" not in data_dict:
            raise ValueError("iterator_shape not found in data_dict")
        if not isinstance(data_dict["iterator_shape"], UdaoEmbedItemShape):
            iterator_shape_dict = data_dict["iterator_shape"]
            data_dict["iterator_shape"] = UdaoEmbedItemShape(
                embedding_input_shape=iterator_shape_dict["embedding_input_shape"],
                feature_names=iterator_shape_dict["feature_names"],
                output_names=iterator_shape_dict["output_names"],
            )
        return cls(**data_dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            k: v if not isinstance(v, UdaoEmbedItemShape) else v.__dict__
            for k, v in self.__dict__.items()
            if v is not None
        }

    def hash(self) -> str:
        elements = [
            str(self.iterator_shape),
            tuple(self.op_groups),
            self.output_size,
            self.lstm_hidden_dim,
            self.lstm_dropout,
            self.readout,
            self.type_embedding_dim,
        ]

        # Conditionally add elements if they are not None
        if self.hist_embedding_dim is not None:
            elements.append(self.hist_embedding_dim)

        if self.bitmap_embedding_dim is not None:
            elements.append(self.bitmap_embedding_dim)

        # Continue adding the rest of the elements
        elements.extend(
            [self.embedding_normalizer, self.n_layers, self.hidden_dim, self.dropout]
        )

        # Convert the list to a tuple, then to string, and encode it
        attributes_tuple = str(tuple(elements)).encode("utf-8")

        sha256_hash = hashlib.sha256(attributes_tuple)
        hex12 = sha256_hash.hexdigest()[:12]
        return "tree_lstm_" + hex12


def get_tree_lstm_mlp(params: TreeLSTMParams) -> UdaoModel:
    model = UdaoModel.from_config(
        embedder_cls=TreeLSTM,
        regressor_cls=MLP,
        iterator_shape=params.iterator_shape,
        embedder_params={
            "output_size": params.output_size,  # 128
            "hidden_dim": params.lstm_hidden_dim,  #
            "dropout": params.lstm_dropout,  # 0.0
            "readout": params.readout,  # "mean"
            "op_groups": params.op_groups,  # ["type", "cbo", "op_enc"]
            "type_embedding_dim": params.type_embedding_dim,  # 8
            "hist_embedding_dim": params.hist_embedding_dim,  # 32
            "bitmap_embedding_dim": params.bitmap_embedding_dim,  # 32
            "embedding_normalizer": params.embedding_normalizer,  # None
        },
        regressor_params={
            "n_layers": params.n_layers,  # 3
            "hidden_dim": params.hidden_dim,  # 512
            "dropout": params.dropout,  # 0.1
        },
    )
    return model


@dataclass
class TreeCNNParams(UdaoParams):
    iterator_shape: UdaoEmbedItemShape
    op_groups: List[str]
    output_size: int = 64
    tcnn_hidden_dim: int = 256
    readout: str = "max"
    type_embedding_dim: int = 8
    hist_embedding_dim: int = 32
    bitmap_embedding_dim: int = 32
    embedding_normalizer: Optional[str] = None
    # MLP
    n_layers: int = 2
    hidden_dim: int = 32
    dropout: float = 0.1

    @classmethod
    def from_dict(cls, data_dict: Dict[str, Any]) -> "TreeCNNParams":
        if "iterator_shape" not in data_dict:
            raise ValueError("iterator_shape not found in data_dict")
        if not isinstance(data_dict["iterator_shape"], UdaoEmbedItemShape):
            iterator_shape_dict = data_dict["iterator_shape"]
            data_dict["iterator_shape"] = UdaoEmbedItemShape(
                embedding_input_shape=iterator_shape_dict["embedding_input_shape"],
                feature_names=iterator_shape_dict["feature_names"],
                output_names=iterator_shape_dict["output_names"],
            )
        return cls(**data_dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            k: v if not isinstance(v, UdaoEmbedItemShape) else v.__dict__
            for k, v in self.__dict__.items()
            if v is not None
        }

    def hash(self) -> str:
        elements = [
            str(self.iterator_shape),
            tuple(self.op_groups),
            self.output_size,
            self.tcnn_hidden_dim,
            self.readout,
            self.type_embedding_dim,
        ]

        # Conditionally add elements if they are not None
        if self.hist_embedding_dim is not None:
            elements.append(self.hist_embedding_dim)

        if self.bitmap_embedding_dim is not None:
            elements.append(self.bitmap_embedding_dim)

        # Continue adding the rest of the elements
        elements.extend(
            [self.embedding_normalizer, self.n_layers, self.hidden_dim, self.dropout]
        )

        # Convert the list to a tuple, then to string, and encode it
        attributes_tuple = str(tuple(elements)).encode("utf-8")

        sha256_hash = hashlib.sha256(attributes_tuple)
        hex12 = sha256_hash.hexdigest()[:12]
        return "tree_cnn_" + hex12


def get_tree_cnn_mlp(params: TreeCNNParams) -> UdaoModel:
    if (
        params.readout != "max"
        or params.output_size != 64
        or params.tcnn_hidden_dim != 256
    ):
        raise ValueError("does not respect the original paper")
    model = UdaoModel.from_config(
        embedder_cls=TreeCNN,
        regressor_cls=MLP,
        iterator_shape=params.iterator_shape,
        embedder_params={
            "output_size": params.output_size,  # 64
            "hidden_dim": params.tcnn_hidden_dim,  # 256
            "readout": params.readout,  # "mean"
            "op_groups": params.op_groups,  # ["type", "cbo", "op_enc"]
            "type_embedding_dim": params.type_embedding_dim,  # 8
            "hist_embedding_dim": params.hist_embedding_dim,  # 32
            "bitmap_embedding_dim": params.bitmap_embedding_dim,  # 32
            "embedding_normalizer": params.embedding_normalizer,  # None
        },
        regressor_params={
            "n_layers": params.n_layers,  # 3
            "hidden_dim": params.hidden_dim,  # 512
            "dropout": params.dropout,  # 0.1
        },
    )
    return model


@dataclass
class QPPNetParams(UdaoParams):
    iterator_shape: UdaoEmbedItemShape
    op_groups: List[str]
    op_node2id: Dict[str, int]

    num_layers: int = 5
    hidden_size: int = 128
    output_size: int = 32
    type_embedding_dim: int = 8
    hist_embedding_dim: int = 32
    bitmap_embedding_dim: int = 32
    embedding_normalizer: Optional[str] = None

    @classmethod
    def from_dict(cls, data_dict: Dict[str, Any]) -> "QPPNetParams":
        if "iterator_shape" not in data_dict:
            raise ValueError("iterator_shape not found in data_dict")
        if not isinstance(data_dict["iterator_shape"], UdaoEmbedItemShape):
            iterator_shape_dict = data_dict["iterator_shape"]
            data_dict["iterator_shape"] = UdaoEmbedItemShape(
                embedding_input_shape=iterator_shape_dict["embedding_input_shape"],
                feature_names=iterator_shape_dict["feature_names"],
                output_names=iterator_shape_dict["output_names"],
            )
        return cls(**data_dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            k: v if not isinstance(v, UdaoEmbedItemShape) else v.__dict__
            for k, v in self.__dict__.items()
            if v is not None
        }

    def hash(self) -> str:
        elements = [
            str(self.iterator_shape),
            tuple(self.op_groups),
            self.output_size,
            self.type_embedding_dim,
        ]

        # Conditionally add elements if they are not None
        if self.hist_embedding_dim is not None:
            elements.append(self.hist_embedding_dim)

        if self.bitmap_embedding_dim is not None:
            elements.append(self.bitmap_embedding_dim)

        # Continue adding the rest of the elements
        elements.extend(
            [
                self.embedding_normalizer,
                self.num_layers,
                self.hidden_size,
            ]
        )

        # Convert the list to a tuple, then to string, and encode it
        attributes_tuple = str(tuple(elements)).encode("utf-8")

        sha256_hash = hashlib.sha256(attributes_tuple)
        hex12 = sha256_hash.hexdigest()[:12]
        return "qppnet_" + hex12


def get_qppnet(params: QPPNetParams) -> UdaoModel:
    model = UdaoModel.from_config(
        embedder_cls=QPPNet,
        regressor_cls=QPPNetOut,
        iterator_shape=params.iterator_shape,
        embedder_params={
            "output_size": params.output_size,  # 128
            "op_groups": params.op_groups,  # ["type", "cbo", "op_enc"]
            "type_embedding_dim": params.type_embedding_dim,  # 8
            "hist_embedding_dim": params.hist_embedding_dim,  # 32
            "bitmap_embedding_dim": params.bitmap_embedding_dim,  # 32
            "embedding_normalizer": params.embedding_normalizer,  # None
            "num_layers": params.num_layers,  # 5
            "hidden_size": params.hidden_size,  # 128
            "op_node2id": params.op_node2id,
        },
        regressor_params={},
    )
    return model


def weights_found(ckp_learning_header: str) -> Optional[str]:
    files = glob.glob(f"{ckp_learning_header}/*.ckpt")
    if len(files) == 0:
        return None
    if len(files) == 1:
        return files[0]
    raise Exception(f"more than one checkpoints {files}")


def checkpoint_learning_params(
    ckp_learning_header: str,
    learning_params: MyLearningParams,
) -> None:
    json_name = "hparams.json"
    if not Path(f"{ckp_learning_header}/{json_name}").exists():
        JsonHandler.dump_to_file(
            learning_params.to_dict(),
            f"{ckp_learning_header}/{json_name}",
            indent=2,
        )
        logger.info(f"saved learning params to {ckp_learning_header}/{json_name}")
    else:
        raise FileExistsError(f"{ckp_learning_header}/{json_name} already exists")


# Model training
def get_tuned_trainer(
    ckp_header: str,
    model: UdaoModel,
    split_iterators: Dict[DatasetType, BaseIterator],
    objectives: List[str],
    params: MyLearningParams,
    device: str,
    num_workers: int = 0,
) -> Tuple[Trainer, UdaoModule, str, bool]:
    ckp_learning_header = f"{ckp_header}/{params.hash()}"
    ckp_weight_path = weights_found(ckp_learning_header)

    tb_logger = TensorBoardLogger(f"tb_logs/{ckp_learning_header}")
    if ckp_weight_path is not None:
        logger.info(f"Model weights found at {ckp_weight_path}, loading...")
        module = UdaoModule.load_from_checkpoint(
            ckp_weight_path,
            model=model,
            objectives=objectives,
            loss=WMAPELoss(),
            metrics=[WeightedMeanAbsolutePercentageError],
            map_location=get_default_device(),
        )
        trainer = pl.Trainer(accelerator=device, logger=tb_logger)
        return trainer, module, ckp_learning_header, True
    logger.info("Model weights not found, training...")

    loss_weights: Optional[Dict[str, float]] = None
    if isinstance(model.embedder, QPPNet):
        loss_weights = {obj: 1.0 if obj == "latency_s" else 0.0 for obj in objectives}
    else:
        if params.loss_weights is not None:
            loss_weights = {obj: w for obj, w in zip(objectives, params.loss_weights)}

    module = UdaoModule(
        model,
        objectives,
        loss=WMAPELoss(),
        learning_params=LearningParams(
            init_lr=params.init_lr,  # 1e-3
            min_lr=params.min_lr,  # 1e-5
            weight_decay=params.weight_decay,  # 1e-2
        ),
        loss_weights=loss_weights,
        metrics=[WeightedMeanAbsolutePercentageError],
    )
    filename_suffix = "-".join(
        [
            "val_loss={val_loss:.4f}",
        ]
        + [
            f"val_{obj}_WMAPE={{val_{obj}_WeightedMeanAbsolutePercentageError:.4f}}"
            for obj in objectives
        ]
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=ckp_learning_header,
        filename="{epoch}-" + filename_suffix,
        auto_insert_metric_name=False,
        verbose=True,
    )
    scheduler = UdaoLRScheduler(setup_cosine_annealing_lr, warmup.UntunedLinearWarmup)
    early_stopping = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=50,
    )
    trainer = pl.Trainer(
        accelerator=device,
        max_steps=params.epochs
        * int(np.ceil(len(split_iterators["train"]) / params.batch_size)),
        max_epochs=params.epochs,
        logger=tb_logger,
        log_every_n_steps=min(len(split_iterators["train"]) // params.batch_size, 50),
        callbacks=[scheduler, checkpoint_callback, early_stopping],
    )
    trainer.fit(
        model=module,
        train_dataloaders=split_iterators["train"].get_dataloader(
            batch_size=params.batch_size,
            num_workers=num_workers,
            shuffle=True,
        ),
        val_dataloaders=split_iterators["val"].get_dataloader(
            batch_size=params.batch_size,
            num_workers=num_workers,
            shuffle=False,
        ),
    )
    best_model_path = checkpoint_callback.best_model_path
    logger.info(f"Best model path: {best_model_path}")
    best_module = UdaoModule.load_from_checkpoint(
        best_model_path,
        model=model,
        objectives=objectives,
        loss=WMAPELoss(),
        metrics=[WeightedMeanAbsolutePercentageError],
        map_location=get_default_device(),
    )
    checkpoint_learning_params(ckp_learning_header, params)
    return trainer, best_module, ckp_learning_header, False


def get_graph_ckp_info(weights_path: str) -> Tuple[str, str, str, str]:
    splits = weights_path.split("/")
    ag_prefix = splits[1]
    model_sign = "_".join(splits[4].split("_")[:2])
    data_processor_path = "/".join(splits[:4] + ["data_processor.pkl"])
    model_params_path = "/".join(splits[:5] + ["model_struct_params.json"])
    return ag_prefix, model_sign, model_params_path, data_processor_path


# Function to add new row for each plan_id
def add_new_rows_for_series(series: pd.Series, fixed_value: int) -> pd.Series:
    df = series.groupby(level="plan_id").size().to_frame("operator_id")
    df["value"] = fixed_value
    new_entries = df.reset_index().set_index(["plan_id", "operator_id"]).value
    new_series = pd.concat([series, new_entries]).sort_index()
    new_series = new_series.loc[series.index.get_level_values("plan_id").unique()]
    return new_series


def add_new_rows_for_df(data: pd.DataFrame, fixed_values: List[float]) -> pd.DataFrame:
    unique_plan_ids = data.index.get_level_values("plan_id").unique()
    df = data.groupby(level="plan_id").size().to_frame("operator_id")
    df[data.columns] = np.array(fixed_values)
    new_entries = df.reset_index().set_index(["plan_id", "operator_id"])
    return pd.concat([data, new_entries]).sort_index().loc[unique_plan_ids]


def get_non_siblings(g: dgl.DGLGraph) -> Dict[int, List[int]]:
    srcs, dsts, eids = g.edges(form="all", order="srcdst")
    child_dep: Dict[int, List[int]] = defaultdict(list)
    for src, dst in zip(srcs.numpy(), dsts.numpy()):
        child_dep[dst].append(src)
    total_nids = set(range(g.num_nodes()))
    non_sibling = {}
    for src, dst, eid in zip(srcs.numpy(), dsts.numpy(), eids.numpy()):
        non_sibling[eid] = total_nids.difference(set(child_dep[dst]))
    return {k: list(v) for k, v in non_sibling.items()}


def get_non_siblings_map(
    dgl_dict: Dict[int, dgl.DGLGraph]
) -> Dict[int, Dict[int, List[int]]]:
    non_siblings_map = {}
    for i, g in dgl_dict.items():
        non_siblings_map[i] = get_non_siblings(g)
    return non_siblings_map


def add_dist_to_graph(g: dgl.DGLGraph) -> dgl.DGLGraph:
    g_nx = dgl.to_networkx(g).to_directed()
    lengths = dict(nx.all_pairs_dijkstra_path_length(g_nx, weight="weight"))

    # Step 2: Initialize g_new with the same number of nodes as g
    g_new = dgl.graph(([], []), num_nodes=g.number_of_nodes())

    # Transfer node features from g to g_new
    for feature_name in g.ndata:
        g_new.ndata[feature_name] = g.ndata[feature_name]

    # Step 3: Add edges and shortest path distances to g_new
    src_list = []
    dst_list = []
    distances = []

    for src, dsts in lengths.items():
        for dst, dist in dsts.items():
            # Skip self-loops
            if src != dst:
                src_list.append(src)
                dst_list.append(dst)
                distances.append(dist)

    # Convert lists to tensors for DGL
    src_tensor = th.tensor(src_list, dtype=th.long)
    dst_tensor = th.tensor(dst_list, dtype=th.long)
    distances_tensor = th.tensor(distances, dtype=th.int32)

    # Add edges to g_new
    g_new.add_edges(src_tensor, dst_tensor)

    # Add edge distances to g_new
    g_new.edata["dist"] = distances_tensor
    return g_new


def add_super_node_to_graph(g: dgl.DGLGraph) -> dgl.DGLGraph:
    # Step 1: Add a super node to the graph, with placeholder
    g.add_nodes(1)

    # Step 2: Connect the super node to all other nodes
    src = th.tensor(list(range(g.number_of_nodes() - 1)), dtype=th.long)
    dst = th.tensor(
        [g.number_of_nodes() - 1] * (g.number_of_nodes() - 1), dtype=th.long
    )
    g.add_edges(src, dst)

    return g


def add_height_to_graph(g: dgl.DGLGraph) -> dgl.DGLGraph:
    # Convert DGL graph to NetworkX graph for easier topological sorting
    nx_g = dgl.to_networkx(g).reverse()

    # Perform topological sort
    topological_order = list(nx.topological_sort(nx_g))  # type: ignore

    # Initialize a dictionary to store heights of nodes
    heights = {node: 0 for node in topological_order}

    # Calculate heights
    for node in topological_order:
        # The height of a node is 1 + the maximum height of its successors
        max_height = 0
        for successor in g.successors(node).tolist():
            max_height = max(max_height, heights[successor])
        heights[node] = max_height + 1

    # Add height information to node features
    # According to the QF paper, height starts from 0
    height_tensor = th.tensor(
        [heights[node] - 1 for node in range(g.number_of_nodes())], dtype=th.int32
    )
    g.ndata["height"] = height_tensor

    return g


def extend_edges_to_graph(g: dgl.DGLGraph) -> dgl.DGLGraph:
    if "height" not in g.ndata:
        g = add_height_to_graph(g)

    # Initialize g_new with the same number of nodes as g
    g_new = dgl.heterograph(
        data_dict={
            ("node", "child", "node"): g.edges(),
            ("node", "descendant", "node"): ([], []),
            ("node", "reachable", "node"): ([], []),
            ("node", "same_height", "node"): ([], []),
        },
        num_nodes_dict={"node": g.number_of_nodes()},
    )
    # Transfer node features from g to g_new
    for feature_name in g.ndata:
        g_new.ndata[feature_name] = g.ndata[feature_name]

    # Step 3: Add edges and shortest path distances to g_new
    g_nx = dgl.to_networkx(g).to_directed()
    lengths = dict(nx.all_pairs_dijkstra_path_length(g_nx, weight="weight"))
    src_list, dst_list, distances = [], [], []

    for src, dsts in lengths.items():
        for dst, dist in dsts.items():
            # Skip self-loops
            if src != dst:
                src_list.append(src)
                dst_list.append(dst)
                distances.append(dist)

    # Convert lists to tensors for DGL
    src_tensor = th.tensor(src_list, dtype=th.long)
    dst_tensor = th.tensor(dst_list, dtype=th.long)
    distances_tensor = th.tensor(distances, dtype=th.int32)

    # Add descendant "edges"
    g_new.add_edges(
        src_tensor, dst_tensor, data={"dist": distances_tensor}, etype="descendant"
    )

    # Add reachable "edges"
    g_new.add_edges(
        src_tensor, dst_tensor, data={"dist": distances_tensor}, etype="reachable"
    )
    g_new.add_edges(
        dst_tensor, src_tensor, data={"dist": -distances_tensor}, etype="reachable"
    )

    # Add same_height "edges"
    height_dict: Dict[int, List[int]] = {}
    for node_i, h_tensor in enumerate(g.ndata["height"]):
        h = int(h_tensor)
        if h not in height_dict:
            height_dict[h] = []
        height_dict[h].append(node_i)
    src_list, dst_list = [], []
    for h, nodes in height_dict.items():
        for node_i in nodes:
            src_list += [node_i] * (len(nodes) - 1)
            dst_list += [node_j for node_j in nodes if node_j != node_i]
    g_new.add_edges(src_tensor, dst_tensor, etype="same_height")
    return g_new


def update_dgl_graphs(
    dgl_dict: Dict[int, QueryPlanStructure],
    funcs: List[Callable[[dgl.DGLGraph], dgl.DGLGraph]],
) -> Dict[int, QueryPlanStructure]:
    new_g_dict = {}
    for i, query in dgl_dict.items():
        new_g_dict[i] = query.graph
        for func in funcs:
            new_g_dict[i] = func(new_g_dict[i])

    for i, new_g in new_g_dict.items():
        dgl_dict[i].graph = new_g

    return dgl_dict


def save_mlp_training_results_one_batch(
    trainer: Trainer,
    module: UdaoModule,
    split_iterators: Dict[DatasetType, BaseIterator],
    params: Namespace,
    ckp_learning_header: str,
    test_results: Dict,
    device: str,
) -> Dict[str, pd.DataFrame]:
    obj_df_dict = {}
    for split, iterator in split_iterators.items():
        if split == "train":
            # remove the random flipping postional encoding augmentation if any.
            iterator.set_augmentations([])
        iterator = cast(QueryPlanIterator, iterator)
        test_file_name = (
            f"obj_df_{split}_with_pred_"
            + "_".join([f"{k}={v:.3f}" for k, v in test_results.items()])
            + f"_{device}"
            + ".pkl"
        )
        if os.path.exists(f"{ckp_learning_header}/{test_file_name}"):
            try:
                cache = PickleHandler.load(ckp_learning_header, test_file_name)
                print("find cached results")
                if not isinstance(cache, Dict) or "obj_df" not in cache:
                    raise KeyError("obj_df not found in cache")
                if not isinstance(cache["obj_df"], pd.DataFrame):
                    raise TypeError("obj_df is not a DataFrame")
                obj_df: pd.DataFrame = cache["obj_df"]
                obj_df_dict[split] = obj_df
                for obj in iterator.objectives.data.columns:
                    print(f"metrics for {obj}: {cache['metrics'][obj]}")
                print(f"time_eval: {cache['time_eval']}")
                continue
            except Exception as e:
                raise e
        else:
            print(f"not found for {split}")
            t1 = time.perf_counter_ns()
            datalaoder = iterator.get_dataloader(
                batch_size=5000,
                num_workers=0 if params.debug else params.num_workers,
                shuffle=False,
            )
            print(f"we have {len(datalaoder)} batches")
            t2 = time.perf_counter_ns()
            pred_list = trainer.predict(model=module, dataloaders=datalaoder)
            if (
                not isinstance(pred_list, list)
                or len(pred_list) == 0
                or not isinstance(pred_list[0], th.Tensor)
            ):
                raise Exception("trainer.predict returns none values")
            pred = th.cat(pred_list).detach().cpu().numpy()  # type: ignore
            t3 = time.perf_counter_ns()

            obj_df = iterator.objectives.data
            obj_names = obj_df.columns.to_list()
            obj_pred_names = [f"{n}_pred" for n in obj_names]
            obj_df[obj_pred_names] = pred
            metrics: Dict[str, Dict[str, float]] = {}
            for obj_ind, obj in enumerate(obj_names):
                metrics[obj] = {}
                y = obj_df[obj].values
                y_pred = pred[:, obj_ind]
                metrics[obj]["wmape"] = local_wmape(y, y_pred)
                metrics[obj]["p50_err"] = local_p50_err(y, y_pred)
                metrics[obj]["p90_err"] = local_p90_err(y, y_pred)
                metrics[obj]["p50_wape"] = local_p50_wape(y, y_pred)
                metrics[obj]["p90_wape"] = local_p90_wape(y, y_pred)
                metrics[obj]["corr"] = float(np.corrcoef(y, y_pred)[0, 1])
            time_eval = {
                "data_prepare_ms": (t2 - t1) / 1e6,
                "pred_ms": (t3 - t2) / 1e6,
            }
            print("metrics: ", metrics)
            print("time_eval: ", time_eval)
            PickleHandler.save(
                {
                    "obj_df": obj_df,
                    "time_eval": time_eval,
                    "metrics": metrics,
                },
                ckp_learning_header,
                test_file_name,
                overwrite=True,
            )
        obj_df_dict[split] = obj_df
    return obj_df_dict


def save_mlp_training_results(
    module: UdaoModule,
    split_iterators: Dict[DatasetType, BaseIterator],
    params: Namespace,
    ckp_learning_header: str,
    test_results: Dict,
    device: str,
) -> Dict[str, pd.DataFrame]:
    obj_df_dict = {}
    local_device = get_default_device()
    print(f"show the local device: {local_device}")
    module.model.to(local_device)
    module.model.eval()
    for p in module.model.parameters():
        p.requires_grad = False

    for split, iterator in split_iterators.items():
        if split == "train":
            # remove the random flipping positional encoding augmentation if any.
            iterator.set_augmentations([])
        iterator = cast(QueryPlanIterator, iterator)
        test_file_name = (
            f"obj_df_{split}_with_"
            + "_".join([f"{k}={v:.3f}" for k, v in test_results.items()])
            + f"_{device}"
            + ".pkl"
        )
        if os.path.exists(f"{ckp_learning_header}/{test_file_name}"):
            try:
                cache = PickleHandler.load(ckp_learning_header, test_file_name)
                print("find cached results")
                if not isinstance(cache, Dict) or "obj_df" not in cache:
                    raise KeyError("obj_df not found in cache")
                if not isinstance(cache["obj_df"], pd.DataFrame):
                    raise TypeError("obj_df is not a DataFrame")
                obj_df: pd.DataFrame = cache["obj_df"]
                obj_df_dict[split] = obj_df
                for obj in iterator.objectives.data.columns:
                    print(f"metrics for {obj}: {cache['metrics'][obj]}")
                print(f"time_eval: {cache['time_eval']}")
                continue
            except Exception as e:
                print("the cache is not ready due to: ", e)
        print(f"not found for {split}, start creating...")
        t1 = time.perf_counter_ns()
        dataloader = iterator.get_dataloader(
            batch_size=5000,
            num_workers=0 if params.debug else params.num_workers,
            shuffle=False,
        )
        t2 = time.perf_counter_ns()
        all_pred = []
        for batch_id, (feature, y) in enumerate(dataloader):
            with th.no_grad():
                feature = cast(QueryPlanInput, feature).to(local_device)
                y_hat = module.model(feature).detach().cpu()
            all_pred.append(y_hat)
            if (batch_id + 1) % 10 == 0:
                print(f"batch {batch_id + 1}/{len(dataloader)} done")
        pred = th.cat(all_pred, dim=0).numpy()
        t3 = time.perf_counter_ns()

        obj_df = iterator.objectives.data
        obj_names = obj_df.columns.to_list()
        obj_pred_names = [f"{n}_pred" for n in obj_names]
        obj_df[obj_pred_names] = pred
        metrics: Dict[str, Dict[str, float]] = {}
        for obj_ind, obj in enumerate(obj_names):
            metrics[obj] = {}
            y = obj_df[obj].values
            y_pred = pred[:, obj_ind]
            metrics[obj]["wmape"] = local_wmape(y, y_pred)
            metrics[obj]["p50_err"] = local_p50_err(y, y_pred)
            metrics[obj]["p90_err"] = local_p90_err(y, y_pred)
            metrics[obj]["p50_wape"] = local_p50_wape(y, y_pred)
            metrics[obj]["p90_wape"] = local_p90_wape(y, y_pred)
            metrics[obj]["corr"] = float(np.corrcoef(y, y_pred)[0, 1])
        time_eval = {
            "data_prepare_ms": (t2 - t1) / 1e6,
            "pred_ms": (t3 - t2) / 1e6,
        }

        print("metrics: ", metrics)
        print("time_eval: ", time_eval)
        PickleHandler.save(
            {
                "obj_df": obj_df,
                "time_eval": time_eval,
                "metrics": metrics,
            },
            ckp_learning_header,
            test_file_name,
            overwrite=True,
        )

    return obj_df_dict


LAT_MIN_MAP = {
    "tpch": {
        "q_compile": {
            "latency_s": 3.073,
            "io_mb": 194.81466484069824,
        },
        "q_all": {
            "latency_s": 0.076,
            "io_mb": 4.1961669921875e-05,
        },
        "qs_lqp_compile": {
            "ana_latency_s": 0.0001625,
            "io_mb": 3.4332275390625e-05,
        },
        "qs_lqp_runtime": {
            "ana_latency_s": 0.0001625,
            "io_mb": 3.4332275390625e-05,
        },
        "qs_pqp_runtime": {
            "ana_latency_s": 0.0001625,
            "io_mb": 3.4332275390625e-05,
        },
    },
    "tpcds": {
        "q_compile": {
            "latency_s": 3.882,
            "io_mb": 2.68405247,
        },
        "q_all": {
            "latency_s": 0.08,
            "io_mb": 1.90734863e-05,
        },
        "qs_lqp_compile": {
            "ana_latency_s": 0.0002,
            "io_mb": 1.90734863e-05,
        },
        "qs_lqp_runtime": {
            "ana_latency_s": 0.0002,
            "io_mb": 1.90734863e-05,
        },
        "qs_pqp_runtime": {
            "ana_latency_s": 0.0002,
            "io_mb": 1.90734863e-05,
        },
    },
}


def calibrate_negative_predictions(
    y_pred: np.ndarray, bm: str, obj: str, q_type: Optional[QType] = None
) -> np.ndarray:
    if q_type is None:
        return np.clip(y_pred, a_min=0, a_max=None)
    else:
        return np.clip(y_pred, a_min=LAT_MIN_MAP[bm][q_type][obj], a_max=None)


def local_wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    sum_abs_error = np.abs(y_pred - y_true).sum()
    sum_scale = np.clip(np.abs(y_true).sum(), a_min=1.17e-06, a_max=None)
    return sum_abs_error / sum_scale


wmape = make_scorer(
    name="WMAPE", score_func=local_wmape, optimum=0.0, greater_is_better=False
)


def local_p50_err(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.percentile(np.abs(y_pred - y_true), 50))


p50_err = make_scorer("p50_err", local_p50_err, optimum=0.0, greater_is_better=False)


def local_p90_err(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.percentile(np.abs(y_pred - y_true), 90))


p90_err = make_scorer("p90_err", local_p90_err, optimum=0.0, greater_is_better=False)


def local_p50_wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return local_p50_err(y_true, y_pred) / np.mean(y_true)


p50_wape = make_scorer("p50_wape", local_p50_wape, optimum=0.0, greater_is_better=False)


def local_p90_wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return local_p90_err(y_true, y_pred) / np.mean(y_true)


p90_wape = make_scorer("p90_wape", local_p90_wape, optimum=0.0, greater_is_better=False)
