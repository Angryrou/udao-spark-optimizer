import hashlib
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

from udao.utils.interfaces import UdaoEmbedItemShape

QType = Literal[
    "q_compile", "q_all", "qs_lqp_compile", "qs_lqp_runtime", "qs_pqp_runtime"
]


@dataclass
class ExtractParams:
    lpe_size: int
    vec_size: int
    seed: int
    q_type: QType
    debug: bool = False

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


def _get_base_parser() -> ArgumentParser:
    # fmt: off
    parser = ArgumentParser(description="Udao Script with Input Arguments")
    # Data-related arguments
    parser.add_argument("--benchmark", type=str, default="tpch",
                        help="Benchmark name")
    parser.add_argument("--q_type", type=str, default="q_compile",
                        choices=["q_compile", "q_all", "qs_lqp_compile",
                                 "qs_lqp_runtime", "qs_pqp_runtime"],
                        help="graph type")
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
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode")
    parser.add_argument("--op_groups", nargs="+", default=["type", "cbo", "op_enc"],
                        help="List of operation groups")
    # fmt: on
    return parser


def get_graph_avg_params() -> Namespace:
    parser = _get_base_parser()
    # fmt: off
    # Embedder parameters
    parser.add_argument("--lpe_size", type=int, default=8,
                        help="Laplacian Positional encoding size (not used)")
    parser.add_argument("--output_size", type=int, default=32,
                        help="Embedder output size")
    parser.add_argument("--type_embedding_dim", type=int, default=8,
                        help="Type embedding dimension")
    parser.add_argument("--vec_size", type=int, default=16,
                        help="Word2Vec embedding size")
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
    parser = _get_base_parser()
    # fmt: off
    # Embedder parameters
    parser.add_argument("--lpe_size", type=int, default=8,
                        help="Laplacian Positional encoding size")
    parser.add_argument("--output_size", type=int, default=32,
                        help="Embedder output size")
    parser.add_argument("--type_embedding_dim", type=int, default=8,
                        help="Type embedding dimension")
    parser.add_argument("--vec_size", type=int, default=16,
                        help="Word2Vec embedding size")
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
