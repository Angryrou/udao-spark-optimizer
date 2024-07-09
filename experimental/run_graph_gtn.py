"""Script to train and evalute a graph embedder + regressor model on a query benchmark."""
from pathlib import Path
from typing import cast

import typer

import torch as th
from udao.data import QueryPlanIterator
from udao.data.utils.query_plan import random_flip_positional_encoding
from udao.model.utils.utils import set_deterministic_torch
from udao.utils.logging import logger
from typer_config import use_yaml_config
from typer_config.callbacks import argument_list_callback

from udao_spark.data.utils import checkpoint_model_structure, get_split_iterators
from udao_spark.model.utils import (
    GraphTransformerMLPParams,
    MyLearningParams,
    get_graph_transformer_mlp,
    get_tuned_trainer,
    save_mlp_training_results,
)
from udao_spark.utils.collaborators import PathWatcher, TypeAdvisor
from udao_spark.utils.params import ExtractParams, get_graph_transformer_params
from udao_trace.utils import JsonHandler

app = typer.Typer(rich_markup_mode="rich",
                  help="CLI to train and evalute a graph embedder + regressor model on a query benchmark.")
logger.setLevel("INFO")


# TODO(glachaud): refactoring in progress
# op_groups was put last because of an issue with lists in typer_config
# see https://maxb2.github.io/typer-config/latest/known_issues/ for details.
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
    embedding_normalizer: str,
    n_layers: int,
    hidden_dim: int,
    dropout: float,
    op_groups: list[str] = typer.Argument(default=None, callback=argument_list_callback)
):
    """Train and evaluate a graph embedder + regressor model on a query benchmark.

    Args:
        benchmark (str): Benchmark name
        scale_factor (int): Scale factor for benchmark
        q_type (str): Level of granularity for query (logical, physical, full query, query stages)
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
        embedding_normalizer (str): Embedding normalizer
        n_layers (int): Number of layers in the regressor
        hidden_dim (int): Hidden dimension of the regressor
        dropout (float): Dropout rate of the regressor
        op_groups (list[str]): List of operation groups (node encodings)
        """
    set_deterministic_torch(seed)
    if benchmark == "tpcds":
        th.set_float32_matmul_precision("medium")  # type: ignore
    device = "gpu" if th.cuda.is_available() else "cpu"
    tensor_dtypes = th.float32
    th.set_default_dtype(tensor_dtypes)  # type: ignore

    # Data definition
    ta = TypeAdvisor(q_type=q_type)
    extract_params = ExtractParams.from_dict(
        {
            "lpe_size": lpe_size,
            "vec_size": vec_size,
            "seed": seed,
            "q_type": q_type,
            "debug": debug,
        }
    )
    pw = PathWatcher(
        Path(__file__).parent, benchmark, debug, extract_params
    )
    split_iterators = get_split_iterators(pw=pw, ta=ta, tensor_dtypes=tensor_dtypes)
    train_iterator = cast(QueryPlanIterator, split_iterators["train"])
    split_iterators["train"].set_augmentations(
        [train_iterator.make_graph_augmentation(random_flip_positional_encoding)]
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
    learning_params = MyLearningParams.from_dict(
        {
            "epochs": epochs,
            "batch_size": batch_size,
            "init_lr": init_lr,
            "min_lr": min_lr,
            "weight_decay": weight_decay,
        }
    )

    model = get_graph_transformer_mlp(model_params)
    tabular_columns = ta.get_tabular_columns()
    objectives = ta.get_objectives()
    logger.info(f"Tabular columns: {tabular_columns}")
    logger.info(f"Objectives: {objectives}")

    ckp_header = checkpoint_model_structure(pw=pw, model_params=model_params)
    trainer, module, ckp_learning_header = get_tuned_trainer(
        ckp_header,
        model,
        split_iterators,
        objectives,
        learning_params,
        device,
        num_workers=0 if debug else num_workers,
        debug=debug,
    )
    test_results = trainer.test(
        model=module,
        dataloaders=split_iterators["test"].get_dataloader(
            batch_size=batch_size,
            num_workers=0 if debug else num_workers,
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
        },
        f"{ckp_learning_header}/test_results.json",
        indent=2,
    )
    # TODO(glachaud): temporary fix
    print(test_results)
    params = {"debug": debug, "num_workers": num_workers}

    obj_df = save_mlp_training_results(
        module=module,
        split_iterators=split_iterators,
        params=params,
        ckp_learning_header=ckp_learning_header,
        test_results=test_results[0],
        device=device,
    )
    
    
if __name__ == "__main__":
    app()
