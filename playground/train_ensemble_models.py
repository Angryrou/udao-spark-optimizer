import os.path
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch as th
from autogluon.tabular import TabularDataset
from udao.data import QueryPlanIterator
from udao.optimization.utils.moo_utils import get_default_device

from udao_spark.model.model_server import ModelServer
from udao_spark.model.mulitlabel_predictor import MultilabelPredictor  # type: ignore
from udao_spark.model.utils import wmape
from udao_spark.utils.collaborators import PathWatcher, TypeAdvisor
from udao_spark.utils.params import ExtractParams, QType, get_ag_parameters
from udao_trace.utils import JsonHandler, ParquetHandler, PickleHandler


def get_graph_embedding(
    ms: ModelServer,
    split_iterators: Dict[str, QueryPlanIterator],
    index_splits: Dict[str, np.ndarray],
    weights_header: str,
) -> Dict[str, np.ndarray]:
    name = "graph_np_dict.pkl"
    try:
        graph_np_dict = PickleHandler.load(weights_header, name)
        print(f"found {weights_header}/{name}")
        if not isinstance(graph_np_dict, Dict):
            raise TypeError(f"graph_np_dict is not a dict: {graph_np_dict}")
        return graph_np_dict
    except FileNotFoundError:
        print("not found, generating...")
    graph_embedding_dict = {}
    bs = 1024
    device = get_default_device()
    for split, iterator in split_iterators.items():
        print(f"start working on {split}")
        n_items = len(index_splits[split])
        dataloader = iterator.get_dataloader(
            batch_size=1024, shuffle=False, num_workers=16
        )
        with th.no_grad():
            all_embeddings = []  # List to store embeddings of all batches
            for batch_id, (batch_input, _) in enumerate(dataloader):
                embedding_input = batch_input.embedding_input
                graph_embedding = ms.model.embedder(embedding_input.to(device))
                all_embeddings.append(
                    graph_embedding
                )  # Append the embeddings of the current batch
                if (batch_id + 1) % 10 == 0:
                    print(f"finished batch {batch_id + 1} / {n_items // bs + 1}")
        # Concatenate all batch embeddings to get the complete embeddings for the split
        graph_embedding_dict[split] = th.cat(all_embeddings, dim=0)
    graph_np_dict = {k: v.cpu().numpy() for k, v in graph_embedding_dict.items()}
    PickleHandler.save(graph_np_dict, weights_header, name)
    return graph_np_dict


def get_ag_data(
    bm: str, q_type: QType, debug: bool, graph_choice: str, weights_path: Optional[str]
) -> Dict:
    ta = TypeAdvisor(q_type=q_type)
    extract_params = ExtractParams.from_dict(  # placeholder
        {
            "lpe_size": 0,
            "vec_size": 0,
            "seed": 0,
            "q_type": q_type,
            "debug": debug,
        }
    )
    pw = PathWatcher(Path(__file__).parent, bm, debug, extract_params)
    df = ParquetHandler.load(pw.cc_prefix, f"df_{ta.get_q_type_for_cache()}.parquet")
    index_splits = PickleHandler.load(
        pw.cc_prefix, f"index_splits_{ta.get_q_type_for_cache()}.pkl"
    )
    if not isinstance(index_splits, Dict):
        raise TypeError(f"index_splits is not a dict: {index_splits}")
    objectives = ta.get_objectives()

    df_splits = {}
    if graph_choice == "none":
        for split, index in index_splits.items():
            df_split = df.loc[index].copy()
            df_split = df_split[ta.get_tabular_columns() + objectives]
            df_splits[split] = df_split
    elif graph_choice in ("avg", "gtn"):
        model_sign = f"graph_{graph_choice}"
        if (
            weights_path is None
            or not os.path.exists(weights_path)
            or len(weights_path.split("/")) != 7
        ):
            raise ValueError(f"weights_path is None: {weights_path}")
        # weights_path = parser.weights_path
        # weights_path = Path(
        #     "cache_and_ckp" /
        #     "tpch_22x2273" /
        #     "q_compile" /
        #     "ea0378f56dcf" /
        #     "graph_avg_0036afb5d8af" /
        #     "learning_b125dbcf7f40" /
        #     "199-val_latency_s_WMAPE=0.148-val_io_mb_WMAPE=0.166.ckpt"
        # )
        header = "/".join(weights_path.split("/")[:4])
        model_params_path = (
            "/".join(weights_path.split("/")[:5]) + "/model_struct_params.json"
        )
        weights_header = "/".join(weights_path.split("/")[:6])
        ms = ModelServer.from_ckp_path(model_sign, model_params_path, weights_path)
        split_iterators = PickleHandler.load(header, "split_iterators.pkl")
        if not isinstance(split_iterators, Dict):
            raise TypeError("split_iterators not found or not a desired type")
        graph_np_dict = get_graph_embedding(
            ms, split_iterators, index_splits, weights_header
        )
        for split, index in index_splits.items():
            df_split = df.loc[index].copy()
            df_split = df_split[ta.get_tabular_columns() + ta.get_objectives()]
            graph_np = graph_np_dict[split]
            ge_dim = graph_np.shape[1]
            ge_cols = [f"ge_{i}" for i in range(ge_dim)]
            df_split[ge_cols] = graph_np
            df_splits[split] = df_split
    else:
        raise ValueError(f"Unknown graph choice: {graph_choice}")

    train_data = TabularDataset(df_splits["train"])
    val_data = TabularDataset(df_splits["val"])
    test_data = TabularDataset(df_splits["test"])
    return {
        "data": [train_data, val_data, test_data],
        "ta": ta,
        "pw": pw,
        "objectives": objectives,
    }


if __name__ == "__main__":
    params = get_ag_parameters().parse_args()
    bm, q_type, debug = params.benchmark, params.q_type, params.debug
    hp_choice, graph_choice = params.hp_choice, params.graph_choice
    num_gpus, ag_sign = params.num_gpus, params.ag_sign
    weights_cache = JsonHandler.load_json("assets/mlp_configs.json")
    try:
        weights_path = weights_cache[bm][hp_choice][graph_choice][q_type]
    except KeyError:
        raise Exception(
            f"weights_path not found for {bm}/{hp_choice}/{graph_choice}/{q_type}"
        )
    ret = get_ag_data(bm, q_type, debug, graph_choice, weights_path)
    train_data, val_data, test_data = ret["data"]
    ta, pw, objectives = ret["ta"], ret["pw"], ret["objectives"]
    if q_type.startswith("qs_"):
        objectives = list(filter(lambda x: x != "latency_s", objectives))
        train_data.drop(columns=["latency_s"], inplace=True)
        val_data.drop(columns=["latency_s"], inplace=True)
        test_data.drop(columns=["latency_s"], inplace=True)
    print("selected features:", train_data.columns)

    # utcnow = datetime.utcnow()
    # timestamp = utcnow.strftime("%Y%m%d_%H%M%S")
    path = "AutogluonModels/{}_{}/{}/{}/{}_{}/".format(
        bm, pw.data_sign, q_type, graph_choice, ag_sign, hp_choice
    )

    if os.path.exists(path):
        predictor = MultilabelPredictor.load(f"{path}")
        print("loaded predictor from", path)
    else:
        print("not found, fitting")
        predictor = MultilabelPredictor(
            path=path,
            labels=objectives,
            problem_types=["regression"] * len(objectives),
            eval_metrics=[wmape] * len(objectives),
            consider_labels_correlation=False,
        )
        predictor.fit(
            train_data=train_data,
            # num_stack_levels=3,
            # num_bag_folds=4,
            # hyperparameters={
            #     "NN_TORCH": {},
            #     "GBM": {},
            #     "CAT": {},
            #     "XGB": {},
            #     "FASTAI": {},
            #     "RF": [
            #         {
            #             "criterion": "gini",
            #             "ag_args": {
            #                 "name_suffix": "Gini",
            #                 "problem_types": ["binary", "multiclass"],
            #             },
            #         },
            #         {
            #             "criterion": "entropy",
            #             "ag_args": {
            #                 "name_suffix": "Entr",
            #                 "problem_types": ["binary", "multiclass"],
            #             },
            #         },
            #         {
            #             "criterion": "squared_error",
            #             "ag_args": {
            #                 "name_suffix": "MSE",
            #                 "problem_types": ["regression", "quantile"],
            #             },
            #         },
            #     ],
            # },
            excluded_model_types=["KNN"],
            tuning_data=val_data,
            # presets='good_quality',
            use_bag_holdout=True,
            num_gpus=num_gpus,
        )

    for obj in predictor.predictors.keys():
        models = predictor.get_predictor(obj).model_names(stack_name="core")
        predictor.get_predictor(obj).fit_weighted_ensemble(
            base_models=[
                m
                for m in models
                if "Large" not in m and "XT" not in m and "ExtraTree" not in m
            ],
            name_suffix="Fast",
        )
        print(
            f"ensemble models for {obj} including "
            f"{predictor.get_predictor(obj).model_names()}"
        )

    # print(path)
    # for obj in objectives:
    #     print(
    #         predictor.get_predictor(obj)
    #         .leaderboard(val_data, extra_metrics=[p90, pearsonr])
    #         .to_string()
    #     )
