import time
from pathlib import Path
from typing import cast

import torch as th
from udao.data import QueryPlanIterator
from udao.model.utils.utils import set_deterministic_torch
from udao.utils.logging import logger

from udao_spark.data.utils import checkpoint_model_structure, get_split_iterators
from udao_spark.model.utils import (
    MyLearningParams,
    TreeLSTMParams,
    get_tree_lstm_mlp,
    get_tuned_trainer,
    save_mlp_training_results,
)
from udao_spark.utils.collaborators import PathWatcher, TypeAdvisor
from udao_spark.utils.params import ExtractParams, get_tree_lstm_params
from udao_trace.utils import JsonHandler

logger.setLevel("INFO")
if __name__ == "__main__":
    params = get_tree_lstm_params().parse_args()
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

    # Model definition and training
    model_params = TreeLSTMParams.from_dict(
        {
            "iterator_shape": split_iterators["train"].shape,
            "op_groups": params.op_groups,
            "output_size": params.output_size,
            "lstm_hidden_dim": params.lstm_hidden_dim,
            "lstm_dropout": params.lstm_dropout,
            "readout": params.readout,
            "type_embedding_dim": params.type_embedding_dim,
            "embedding_normalizer": params.embedding_normalizer,
            "n_layers": params.n_layers,
            "hidden_dim": params.hidden_dim,
            "dropout": params.dropout,
        }
    )
    learning_params = MyLearningParams.from_dict(
        {
            "epochs": params.epochs,
            "batch_size": params.batch_size,
            "init_lr": params.init_lr,
            "min_lr": params.min_lr,
            "weight_decay": params.weight_decay,
        }
    )

    model = get_tree_lstm_mlp(model_params)
    # prepare the model structure path
    tabular_columns = ta.get_tabular_columns()
    objectives = ta.get_objectives()
    logger.info(f"Tabular columns: {tabular_columns}")
    logger.info(f"Objectives: {objectives}")

    ckp_header = checkpoint_model_structure(pw=pw, model_params=model_params)
    start_time = time.perf_counter_ns()
    trainer, module, ckp_learning_header = get_tuned_trainer(
        ckp_header,
        model,
        split_iterators,
        objectives,
        learning_params,
        device,
        num_workers=0 if params.debug else params.num_workers,
    )
    train_time_secs = (time.perf_counter_ns() - start_time) / 1e9
    test_results = trainer.test(
        model=module,
        dataloaders=split_iterators["test"].get_dataloader(
            batch_size=params.batch_size,
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
    obj_df = save_mlp_training_results(
        module=module,
        split_iterators=split_iterators,
        params=params,
        ckp_learning_header=ckp_learning_header,
        test_results=test_results[0],
        device=device,
    )
