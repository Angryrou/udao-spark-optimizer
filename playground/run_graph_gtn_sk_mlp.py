import hashlib
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import torch as th
from udao.data import QueryPlanIterator
from udao.data.utils.query_plan import random_flip_positional_encoding
from udao.model import UdaoModel
from udao.model.embedders.layers.multi_head_attention import AttentionLayerName
from udao.model.utils.utils import set_deterministic_torch
from udao.utils.interfaces import UdaoEmbedItemShape
from udao.utils.logging import logger

from udao_spark.data.utils import get_split_iterators
from udao_spark.model.embedders.graph_transformer import GraphTransformer
from udao_spark.model.regressors.sk_mlp import SkipConnectionMLP
from udao_spark.model.utils import MyLearningParams, train_and_dump
from udao_spark.utils.collaborators import PathWatcher, TypeAdvisor
from udao_spark.utils.params import (
    ExtractParams,
    UdaoParams,
    get_graph_transformer_params,
)


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


def get_params() -> ArgumentParser:
    parser = get_graph_transformer_params()
    # fmt: off
    parser.add_argument("--activate", type=str, default="relu",
                        choices=["relu", "elu", "tanh"],
                        help="Activation function.")
    parser.add_argument("--use_batchnorm", action="store_true",
                        help="Whether to use batch normalization.")
    # fmt: on
    return parser


logger.setLevel("INFO")
if __name__ == "__main__":
    params = get_params().parse_args()
    set_deterministic_torch(params.seed)
    if params.benchmark == "tpcds":
        th.set_float32_matmul_precision("medium")  # type: ignore
    print(params)
    device = "gpu" if th.cuda.is_available() else "cpu"
    tensor_dtypes = th.float32
    th.set_default_dtype(tensor_dtypes)  # type: ignore

    # Data definition
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
    )
    split_iterators = get_split_iterators(pw=pw, ta=ta, tensor_dtypes=tensor_dtypes)
    train_iterator = cast(QueryPlanIterator, split_iterators["train"])
    split_iterators["train"].set_augmentations(
        [train_iterator.make_graph_augmentation(random_flip_positional_encoding)]
    )
    # Model definition and training
    model_params = GraphTransformerSKMLPParams.from_dict(
        {
            "iterator_shape": split_iterators["train"].shape,
            "op_groups": params.op_groups,
            "output_size": params.output_size,
            "pos_encoding_dim": params.pos_encoding_dim,
            "gtn_n_layers": params.gtn_n_layers,
            "gtn_n_heads": params.gtn_n_heads,
            "readout": params.readout,
            "type_embedding_dim": params.type_embedding_dim,
            "hist_embedding_dim": params.hist_embedding_dim,
            "bitmap_embedding_dim": params.bitmap_embedding_dim,
            "embedding_normalizer": params.embedding_normalizer,
            "gtn_dropout": params.gtn_dropout,
            "n_layers": params.n_layers,
            "hidden_dim": params.hidden_dim,
            "dropout": params.dropout,
            "use_batchnorm": params.use_batchnorm,
            "activate": params.activate,
        }
    )

    if params.loss_weights is not None:
        if len(params.loss_weights) != len(ta.get_objectives()):
            raise ValueError(
                f"loss_weights must have the same length as objectives, "
                f"got {len(params.loss_weights)} and {len(ta.get_objectives())}"
            )

    learning_params = MyLearningParams.from_dict(
        {
            "epochs": params.epochs,
            "batch_size": params.batch_size,
            "init_lr": params.init_lr,
            "min_lr": params.min_lr,
            "weight_decay": params.weight_decay,
            "loss_weights": params.loss_weights,
        }
    )

    model = get_graph_transformer_sk_mlp(model_params)

    train_and_dump(
        ta=ta,
        pw=pw,
        model=model,
        split_iterators=split_iterators,
        extract_params=extract_params,
        model_params=model_params,
        learning_params=learning_params,
        params=params,
        device=device,
    )
