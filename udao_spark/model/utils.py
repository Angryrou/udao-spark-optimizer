import glob
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import lightning.pytorch as pl
import numpy as np
import pytorch_warmup as warmup
from autogluon.core.metrics import make_scorer
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torchmetrics import WeightedMeanAbsolutePercentageError
from udao.data import BaseIterator
from udao.data.utils.utils import DatasetType
from udao.model import MLP, GraphAverager, GraphTransformer, UdaoModel, UdaoModule
from udao.model.module import LearningParams
from udao.model.utils.losses import WMAPELoss
from udao.model.utils.schedulers import UdaoLRScheduler, setup_cosine_annealing_lr
from udao.utils.interfaces import UdaoEmbedItemShape
from udao.utils.logging import logger

from udao_trace.utils import JsonHandler

from ..utils.params import UdaoParams


@dataclass
class GraphAverageMLPParams(UdaoParams):
    iterator_shape: UdaoEmbedItemShape
    op_groups: List[str]
    output_size: int = 32
    type_embedding_dim: int = 8
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
        attributes_tuple = str(
            (
                str(self.iterator_shape),
                tuple(self.op_groups),
                self.output_size,
                self.type_embedding_dim,
                self.embedding_normalizer,
                self.n_layers,
                self.hidden_dim,
                self.dropout,
            )
        ).encode("utf-8")
        sha256_hash = hashlib.sha256(attributes_tuple)
        hex12 = sha256_hash.hexdigest()[:12]
        return "graph_avg_" + hex12


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
    embedding_normalizer: Optional[str] = None
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
                self.embedding_normalizer,
                self.n_layers,
                self.hidden_dim,
                self.dropout,
            )
        ).encode("utf-8")
        sha256_hash = hashlib.sha256(attributes_tuple)
        hex12 = sha256_hash.hexdigest()[:12]
        return "graph_gtn_" + hex12


@dataclass
class MyLearningParams(UdaoParams):
    epochs: int = 2
    batch_size: int = 512
    init_lr: float = 1e-1
    min_lr: float = 1e-5
    weight_decay: float = 1e-2

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
        return "learning_" + hex12


def get_graph_avg_mlp(params: GraphAverageMLPParams) -> UdaoModel:
    model = UdaoModel.from_config(
        embedder_cls=GraphAverager,
        regressor_cls=MLP,
        iterator_shape=params.iterator_shape,
        embedder_params={
            "output_size": params.output_size,  # 32
            "op_groups": params.op_groups,  # ["type", "cbo", "op_enc"]
            "type_embedding_dim": params.type_embedding_dim,  # 8
            "embedding_normalizer": params.embedding_normalizer,  # None
        },
        regressor_params={
            "n_layers": params.n_layers,  # 3
            "hidden_dim": params.hidden_dim,  # 512
            "dropout": params.dropout,  # 0.1
        },
    )
    return model


def get_graph_gtn_mlp(params: GraphTransformerMLPParams) -> UdaoModel:
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
            "embedding_normalizer": params.embedding_normalizer,  # None
        },
        regressor_params={
            "n_layers": params.n_layers,  # 3
            "hidden_dim": params.hidden_dim,  # 512
            "dropout": params.dropout,  # 0.1
        },
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
) -> Tuple[Trainer, UdaoModule, str]:
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
        )
        trainer = pl.Trainer(accelerator=device, logger=tb_logger)
        return trainer, module, ckp_learning_header
    logger.info("Model weights not found, training...")
    module = UdaoModule(
        model,
        objectives,
        loss=WMAPELoss(),
        learning_params=LearningParams(
            init_lr=params.init_lr,  # 1e-3
            min_lr=params.min_lr,  # 1e-5
            weight_decay=params.weight_decay,  # 1e-2
        ),
        metrics=[WeightedMeanAbsolutePercentageError],
    )
    filename_suffix = "-".join(
        [
            f"val_{obj}_WMAPE={{val_{obj}_WeightedMeanAbsolutePercentageError:.3f}}"
            for obj in objectives
        ]
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckp_learning_header,
        filename="{epoch}-" + filename_suffix,
        auto_insert_metric_name=False,
    )
    scheduler = UdaoLRScheduler(setup_cosine_annealing_lr, warmup.UntunedLinearWarmup)
    trainer = pl.Trainer(
        accelerator=device,
        max_epochs=params.epochs,
        logger=tb_logger,
        callbacks=[scheduler, checkpoint_callback],
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
    checkpoint_learning_params(ckp_learning_header, params)
    return trainer, module, ckp_learning_header


def get_graph_ckp_info(weights_path: str) -> Tuple[str, str, str, str]:
    splits = weights_path.split("/")
    ag_prefix = splits[1]
    model_sign = "_".join(splits[4].split("_")[:2])
    data_processor_path = "/".join(splits[:4] + ["data_processor.pkl"])
    model_params_path = "/".join(splits[:5] + ["model_struct_params.json"])
    return ag_prefix, model_sign, model_params_path, data_processor_path


def local_wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    sum_abs_error = np.abs(y_pred - y_true).sum()
    sum_scale = np.abs(y_true).sum()
    return sum_abs_error / np.max([sum_scale, 1.17e-06])


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
