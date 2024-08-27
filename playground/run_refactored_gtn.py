from pathlib import Path
from typing import Optional, cast

import torch as th
import typer
from typer_config import use_yaml_config
from typer_config.callbacks import argument_list_callback
from typing_extensions import Annotated
from udao.data import QueryPlanIterator
from udao.data.utils.query_plan import random_flip_positional_encoding
from udao.model import UdaoModel
from udao.model.regressors import MLP
from udao.model.utils.utils import set_deterministic_torch
from udao.utils.logging import logger

from udao_spark.data.utils import get_split_iterators
from udao_spark.model.embedders.transformer import model_factory
from udao_spark.model.utils import (
    GraphTransformerMLPParams,
    MyLearningParams,
    train_and_dump,
)
from udao_spark.utils.collaborators import PathWatcher, TypeAdvisor
from udao_spark.utils.params import ExtractParams

app = typer.Typer(
    rich_markup_mode="rich",
    help="CLI to train and evalute a graph embedder + regressor model on a query benchmark.",  # noqa: E501
)


@app.command()
@use_yaml_config(default_value="configs/default_config.yaml")
def main(
    benchmark: str,
    scale_factor: int,
    q_type: str,
    debug: bool,
    seed: int,
    lpe_size: int,
    vec_size: int,
    init_lr: float,
    min_lr: float,
    weight_decay: float,
    epochs: int,
    batch_size: int,
    num_workers: int,
    output_size: int,
    pos_encoding_dim: int,
    gtn_n_layers: int,
    gtn_n_heads: int,
    readout: str,
    type_embedding_dim: int,
    hist_embedding_dim: int,
    bitmap_embedding_dim: int,
    n_layers: int,
    hidden_dim: int,
    dropout: float,
    fold: Annotated[Optional[int], typer.Argument()] = None,
    embedding_normalizer: Annotated[Optional[str], typer.Argument()] = None,
    op_groups: list[str] = typer.Argument(
        default=None, callback=argument_list_callback
    ),
    loss_weights: Annotated[Optional[float], typer.Argument()] = None,
) -> None:
    """Train and evaluate a graph embedder + regressor model on a query benchmark.

    Args:
        benchmark (str): Benchmark name
        scale_factor (int): Scale factor for benchmark
        q_type (str): Level of granularity for query
            (logical, physical, full query, query stages)
        debug (bool): Debug mode
        seed (int): Random seed
        lpe_size (int): Size of Laplacian Positional Encoding
        vec_size (int): Word2Vec embedding size
        init_lr (float): Initial learning rate
        min_lr (float): Minimum learning rate
        weight_decay (float): Weight decay
        epochs (int): Number of training epochs
        batch_size (int): Batch size
        num_workers (int): Number of workers (only in non-debug)
        output_size (int): Embedder output size
        pos_encoding_dim (int): Size of Positional Encoding
        gtn_n_layers (int): Number of layers in the GTN
        gtn_n_heads (int): Number of heads in the GTN
        readout (str): Readout function ("mean", "max", or "sum")
        type_embedding_dim (int): Type embedding dimension
        n_layers (int): Number of layers in the regressor
        hidden_dim (int): Hidden dimension of the regressor
        dropout (float): Dropout rate of the regressor
        fold (int): Fold number (from 1 to 10)
        embedding_normalizer (str): Embedding normalizer
        op_groups (list[str]): List of operation groups (node encodings)
        loss_weights (float): TODO: fill the doc
    """
    set_deterministic_torch(seed)
    if benchmark == "tpcds":
        th.set_float32_matmul_precision("medium")  # type: ignore
    device = "gpu" if th.cuda.is_available() else "cpu"
    tensor_dtypes = th.float32
    th.set_default_dtype(tensor_dtypes)  # type: ignore

    # Data definition
    ta = TypeAdvisor(q_type=q_type)  # type: ignore
    extract_params = ExtractParams.from_dict(
        {
            "lpe_size": lpe_size,
            "vec_size": vec_size,
            "seed": seed,
            "q_type": q_type,
            "debug": debug,
        }
    )
    pw = PathWatcher(Path(__file__).parent, benchmark, debug, extract_params, fold)
    split_iterators = get_split_iterators(pw=pw, ta=ta, tensor_dtypes=tensor_dtypes)
    train_iterator = cast(QueryPlanIterator, split_iterators["train"])
    split_iterators["train"].set_augmentations(
        [train_iterator.make_graph_augmentation(random_flip_positional_encoding)]
    )

    # TODO: extract the inputs that I need from this function
    iterator_shape = split_iterators["train"].shape
    embedding_input_shapes = iterator_shape.embedding_input_shape
    input_size = sum(
        [embedding_input_shapes[name] for name in op_groups if name != "type"]
    )
    n_op_types = None
    if "type" in op_groups:
        n_op_types = iterator_shape.embedding_input_shape["type"]

    # TODO: create graph transformer using the model factory
    graph_transformer = model_factory.create_graph_transformer(
        in_dim=input_size,
        hidden_dim=hidden_dim,
        output_size=output_size,
        n_layers=gtn_n_layers,
        n_heads=gtn_n_heads,
        n_op_types=n_op_types,
        op_groups=op_groups,
        type_embedding_dim=type_embedding_dim,
        hist_embedding_dim=hist_embedding_dim,
        bitmap_embedding_dim=bitmap_embedding_dim,
        pos_encoding_dim=lpe_size,
    )

    model_params = GraphTransformerMLPParams.from_dict(
        {
            "iterator_shape": split_iterators["train"].shape,
            "op_groups": op_groups,
            "output_size": output_size,
            "pos_encoding_dim": pos_encoding_dim,
            "gtn_n_layers": gtn_n_layers,
            "gtn_n_heads": gtn_n_heads,
            "readout": readout,
            "type_embedding_dim": type_embedding_dim,
            "embedding_normalizer": embedding_normalizer,
            "n_layers": n_layers,
            "hidden_dim": hidden_dim,
            "dropout": dropout,
        }
    )

    mlp_params = MLP.Params(
        input_embedding_dim=output_size,
        input_features_dim=len(iterator_shape.feature_names),
        output_dim=len(iterator_shape.output_names),
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )

    mlp = MLP(mlp_params)

    model = UdaoModel(graph_transformer, mlp)  # type: ignore
    # Model definition and training
    # model_params = GraphTransformerMLPParams.from_dict(
    #     {
    #         "iterator_shape": split_iterators["train"].shape,
    #         "op_groups": op_groups,
    #         "output_size": output_size,
    #         "pos_encoding_dim": pos_encoding_dim,
    #         "gtn_n_layers": gtn_n_layers,
    #         "gtn_n_heads": gtn_n_heads,
    #         "readout": readout,
    #         "type_embedding_dim": type_embedding_dim,
    #         "embedding_normalizer": embedding_normalizer,
    #         "n_layers": n_layers,
    #         "hidden_dim": hidden_dim,
    #         "dropout": dropout,
    #     }
    # )

    if loss_weights is not None:
        if len(loss_weights) != len(ta.get_objectives()):  # type: ignore
            raise ValueError(
                f"loss_weights must have the same length as objectives, "
                f"got {len(loss_weights)} and {len(ta.get_objectives())}"  # type: ignore
            )

    learning_params = MyLearningParams.from_dict(
        {
            "epochs": epochs,
            "batch_size": batch_size,
            "init_lr": init_lr,
            "min_lr": min_lr,
            "weight_decay": weight_decay,
            "loss_weights": loss_weights,
        }
    )

    # TODO (glachaud): temporary fix before the refactor is completed.
    # TODO (glachaud): changed from dict to dotdict to allow member access
    # see dotdict below.
    params = dotdict()
    params.debug = debug  # type: ignore
    params.num_workers = num_workers  # type: ignore
    train_and_dump(
        ta=ta,
        pw=pw,
        model=model,
        split_iterators=split_iterators,
        extract_params=extract_params,
        model_params=model_params,  # type: ignore
        learning_params=learning_params,
        params=params,  # type: ignore
        device=device,
    )


# TODO(glachaud): temporary fix for params bug
# taken from https://stackoverflow.com/a/23689767
class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__  # type: ignore
    __delattr__ = dict.__delitem__  # type: ignore


logger.setLevel("INFO")
if __name__ == "__main__":
    app()
