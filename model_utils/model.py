import glob
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lightning.pytorch as pl
import pytorch_warmup as warmup
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torchmetrics import WeightedMeanAbsolutePercentageError
from udao.data import BaseIterator
from udao.data.utils.utils import DatasetType
from udao.model import MLP, GraphAverager, UdaoModel, UdaoModule
from udao.model.module import LearningParams
from udao.model.utils.losses import WMAPELoss
from udao.model.utils.schedulers import UdaoLRScheduler, setup_cosine_annealing_lr
from udao.utils.interfaces import UdaoEmbedItemShape
from udao.utils.logging import logger

from udao_trace.utils import JsonHandler, PickleHandler

from .utils import PathWatcher


@dataclass
class GraphAverageMLPParams:
    iterator_shape: UdaoEmbedItemShape
    op_groups: List[str]
    output_size: int = 32
    type_embedding_dim: int = 8
    embedding_normalizer: Optional[str] = None
    n_layers: int = 2
    hidden_dim: int = 32
    dropout: float = 0.1

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
class MyLearningParams:
    epochs: int = 2
    batch_size: int = 512
    init_lr: float = 1e-1
    min_lr: float = 1e-5
    weight_decay: float = 1e-2

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
    return UdaoModel.from_config(
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


def weights_found(ckp_learning_header: str) -> Optional[str]:
    files = glob.glob(f"{ckp_learning_header}/*.ckpt")
    if len(files) == 0:
        return None
    if len(files) == 1:
        return files[0]
    raise Exception(f"more than one checkpoints {files}")


def checkpoint_learning_params(
    ckp_header: str,
    learning_params: MyLearningParams,
) -> None:
    ckp_header += f"/{learning_params.hash()}"
    name = "hparams.pkl"
    json_name = "hparams.json"
    if not Path(f"{ckp_header}/{name}").exists():
        PickleHandler.save(learning_params, ckp_header, name)
        JsonHandler.dump_to_file(
            learning_params.__dict__,
            f"{ckp_header}/{json_name}",
            indent=2,
        )
        logger.info(f"saved learning params to {ckp_header}/{name}")
    else:
        raise FileExistsError(f"{ckp_header}/{name} already exists")


def checkpoint_model_structure(
    pw: PathWatcher, model_params: GraphAverageMLPParams
) -> str:
    model_struct_hash = model_params.hash()
    ckp_header = f"{pw.cc_extract_prefix}/{model_struct_hash}"
    name = "model_struct_params.pkl"
    json_name = "model_struct_params.json"
    if not Path(f"{ckp_header}/{name}").exists():
        Path(ckp_header).mkdir(parents=True, exist_ok=True)
        PickleHandler.save(model_params, ckp_header, name)
        JsonHandler.dump_to_file(
            model_params.to_dict(),
            f"{ckp_header}/{json_name}",
            indent=2,
        )
        logger.info(f"saved model structure params to {ckp_header}")
    else:
        logger.info(f"found {ckp_header}/{name}")
    return ckp_header


# Model training
def get_tuned_trainer(
    ckp_header: str,
    model: UdaoModel,
    split_iterators: Dict[DatasetType, BaseIterator],
    objectives: List[str],
    params: MyLearningParams,
    device: str,
    num_workers: int = 0,
) -> Tuple[Trainer, UdaoModule]:
    ckp_learning_header = f"{ckp_header}/{params.hash()}"
    ckp_weight_path = weights_found(ckp_learning_header)
    tb_logger = TensorBoardLogger("tb_logs")
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
        return trainer, module
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

    checkpoint_callback = ModelCheckpoint(
        dirpath=ckp_learning_header,
        filename="{epoch}"
        "-val_lat_WMAPE={val_latency_s_WeightedMeanAbsolutePercentageError:.3f}"
        "-val_io_WMAPE={val_io_mb_WeightedMeanAbsolutePercentageError:.3f}",
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
    checkpoint_learning_params(ckp_header, params)
    return trainer, module
