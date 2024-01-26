import hashlib
from abc import ABC
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from typing import Any, Dict, Literal

QType = Literal[
    "q_compile", "q_all", "qs_lqp_compile", "qs_lqp_runtime", "qs_pqp_runtime"
]


@dataclass
class UdaoParams(ABC):
    def hash(self) -> str:
        raise NotImplementedError

    def to_dict(self) -> dict:
        return self.__dict__


@dataclass
class ExtractParams(UdaoParams):
    lpe_size: int
    vec_size: int
    seed: int
    q_type: QType
    debug: bool = False

    @classmethod
    def from_dict(cls, data_dict: Dict[str, Any]) -> "ExtractParams":
        return cls(**data_dict)

    def hash(self) -> str:
        attributes_tuple = str(
            (
                self.lpe_size,
                self.vec_size,
                self.seed,
            )
        ).encode("utf-8")
        sha256_hash = hashlib.sha256(attributes_tuple)
        hex12 = self.q_type + "/" + sha256_hash.hexdigest()[:12]
        if self.debug:
            return hex12 + "_debug"
        return hex12


def get_base_parser() -> ArgumentParser:
    # fmt: off
    parser = ArgumentParser(description="Udao Script with Input Arguments")
    # Data-related arguments
    parser.add_argument("--benchmark", type=str, default="tpch",
                        help="Benchmark name")
    parser.add_argument("--q_type", type=str, default="q_compile",
                        choices=["q_compile", "q_all", "qs_lqp_compile",
                                 "qs_lqp_runtime", "qs_pqp_runtime"],
                        help="graph type")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode")
    # fmt: on
    return parser


def _get_graph_base_parser() -> ArgumentParser:
    # fmt: off
    parser = get_base_parser()
    # Common embedding parameters
    parser.add_argument("--lpe_size", type=int, default=8,
                        help="Provided Laplacian Positional encoding size")
    parser.add_argument("--vec_size", type=int, default=16,
                        help="Word2Vec embedding size")
    # Learning parameters
    parser.add_argument("--init_lr", type=float, default=1e-1,
                        help="Initial learning rate")
    parser.add_argument("--min_lr", type=float, default=1e-5,
                        help="Minimum learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-2,
                        help="Weight decay")
    parser.add_argument("--epochs", type=int, default=2,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=512,
                        help="Batch size")
    # Others
    parser.add_argument("--num_workers", type=int, default=15,
                        help="non-debug only")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--op_groups", nargs="+", default=["type", "cbo", "op_enc"],
                        help="List of operation groups")
    # fmt: on
    return parser


def get_graph_avg_params() -> Namespace:
    parser = _get_graph_base_parser()
    # fmt: off
    # Embedder parameters
    parser.add_argument("--output_size", type=int, default=32,
                        help="Embedder output size")
    parser.add_argument("--type_embedding_dim", type=int, default=8,
                        help="Type embedding dimension")
    parser.add_argument("--embedding_normalizer", type=str, default=None,
                        help="Embedding normalizer")
    # Regressor parameters
    parser.add_argument("--n_layers", type=int, default=2,
                        help="Number of layers in the regressor")
    parser.add_argument("--hidden_dim", type=int, default=32,
                        help="Hidden dimension of the regressor")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate")
    # fmt: on
    return parser.parse_args()


def get_graph_gtn_params() -> Namespace:
    parser = _get_graph_base_parser()
    # fmt: off
    # Embedder parameters
    parser.add_argument("--output_size", type=int, default=32,
                        help="Embedder output size")
    parser.add_argument("--pos_encoding_dim", type=int, default=8,
                        help="Positional encoding dimension for use")
    parser.add_argument("--gtn_n_layers", type=int, default=2,
                        help="Number of layers in the GTN")
    parser.add_argument("--gtn_n_heads", type=int, default=2,
                        help="Number of heads in the GTN")
    parser.add_argument("--readout", type=str, default="mean",
                        choices=["mean", "max", "sum"],
                        help="Readout function")
    parser.add_argument("--type_embedding_dim", type=int, default=8,
                        help="Type embedding dimension")
    parser.add_argument("--embedding_normalizer", type=str, default=None,
                        help="Embedding normalizer")
    # Regressor parameters
    parser.add_argument("--n_layers", type=int, default=2,
                        help="Number of layers in the regressor")
    parser.add_argument("--hidden_dim", type=int, default=32,
                        help="Hidden dimension of the regressor")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate")
    # fmt: on
    return parser.parse_args()


def get_ag_parameters() -> Namespace:
    parser = get_base_parser()
    # fmt: off
    parser.add_argument("--graph_choice", type=str, default="none",
                        choices=["none", "avg", "gtn"])
    # fmt: on

    return parser.parse_args()
