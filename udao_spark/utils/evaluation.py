import os
import time
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch as th
from autogluon.core import TabularDataset
from autogluon.tabular import TabularPredictor
from udao.data import QueryPlanIterator
from udao.optimization.utils.moo_utils import get_default_device

from udao_spark.data.extractors.injection_extractor import (
    get_non_decision_inputs_for_q_compile,
)
from udao_spark.model.model_server import AGServer, ModelServer
from udao_spark.model.utils import (
    calibrate_negative_predictions,
    local_p50_err,
    local_p50_wape,
    local_p90_err,
    local_p90_wape,
    local_wmape,
)
from udao_spark.optimizer.atomic_optimizer import AtomicOptimizer
from udao_spark.optimizer.utils import get_ag_meta
from udao_spark.utils.collaborators import PathWatcher, TypeAdvisor, get_data_sign
from udao_spark.utils.params import ExtractParams, QType
from udao_trace.configuration import SparkConf
from udao_trace.utils import BenchmarkType, ParquetHandler, PickleHandler
from udao_trace.workload import Benchmark


def extract_non_decision_df(non_decision_input_dict: Dict) -> pd.DataFrame:
    """
    extract the non_decision dict to a DataFrame
    """
    df = pd.DataFrame.from_dict(non_decision_input_dict, orient="index")
    df = df.reset_index().rename(columns={"index": "id"})
    df["id"] = df["id"].str.split("-").str[-1].astype(int)
    df.set_index("id", inplace=True, drop=False)
    df.sort_index(inplace=True)
    return df


def get_non_decision_inputs(
    base_dir: Path,
    params: Namespace,
    decision_vars: List[str],
    cache_file: str = "non_decision_df_03-07.parquet",
) -> Tuple[pd.DataFrame, AtomicOptimizer]:
    # prepare parameters
    bm, q_type = params.benchmark, params.q_type
    hp_choice, graph_choice = params.hp_choice, params.graph_choice
    ag_sign = params.ag_sign
    il, bs, tl = params.infer_limit, params.infer_limit_batch_size, params.ag_time_limit
    ag_meta = get_ag_meta(bm, hp_choice, graph_choice, q_type, ag_sign, il, bs, tl)
    ag_full_name = ag_meta["ag_full_name"]
    cache_header = (
        f"robustness_eval/violation/{bm}/{q_type}/{graph_choice}/{ag_full_name}"
    )
    spark_conf = SparkConf(str(base_dir / "assets/spark_configuration_aqe_on.json"))
    # use functions in HierarchicalOptimizer to extract the non-decision inputs
    atomic_optimizer = AtomicOptimizer(
        bm=bm,
        model_sign=ag_meta["model_sign"],
        graph_model_params_path=ag_meta["model_params_path"],
        graph_weights_path=ag_meta["graph_weights_path"],
        q_type=q_type,
        data_processor_path=ag_meta["data_processor_path"],
        spark_conf=spark_conf,
        decision_variables=decision_vars,
        ag_path=ag_meta["ag_path"],
        clf_json_path=None
        if params.disable_failure_clf
        else str(base_dir / f"assets/{bm}_valid_clf_meta.json"),
        clf_recall_xhold=params.clf_recall_xhold,
        verbose=False,
    )

    try:
        df = ParquetHandler.load(cache_header, cache_file)
        atomic_optimizer.ag_ms = AGServer.from_ckp_path(
            model_sign=ag_meta["model_sign"],
            graph_model_params_path=ag_meta["model_params_path"],
            graph_weights_path=ag_meta["graph_weights_path"],
            q_type=q_type,
            ag_path=ag_meta["ag_path"],
            clf_json_path=str(base_dir / f"assets/{bm}_valid_clf_meta.json"),
            clf_recall_xhold=params.clf_recall_xhold,
        )
        print("found cached non_decision_df...")
        return df, atomic_optimizer
    except FileNotFoundError:
        print("no cached non_decision_df found, generating...")
    except Exception as e:
        print(f"error loading cached non_decision_df: {e}")
        raise e

    # prepare the traces
    sample_header = str(base_dir / "assets/query_plan_samples")
    if bm == "tpch":
        benchmark = Benchmark(BenchmarkType.TPCH, params.scale_factor)
        raw_traces = [
            f"{sample_header}/{bm}/{bm}100_{q}-1_1,1g,16,16,48m,200,true,0.6,"
            f"64MB,0.2,0MB,10MB,200,256MB,5,128MB,4MB,0.2,1024KB"
            f"_application_1701736595646_{2557 + i}.json"
            for i, q in enumerate(benchmark.templates)
        ]
    elif bm == "tpcds":
        benchmark = Benchmark(BenchmarkType.TPCDS, params.scale_factor)
        raw_traces = [
            f"{sample_header}/{bm}/{bm}100_{q}-1_1,1g,16,16,48m,200,true,0.6,"
            f"64MB,0.2,0MB,10MB,200,256MB,5,128MB,4MB,0.2,1024KB"
            f"_application_1701737506122_{3283 + i}.json"
            for i, q in enumerate(benchmark.templates)
        ]
    else:
        raise ValueError(f"benchmark {bm} is not supported")
    for trace in raw_traces:
        print(trace)
        if not Path(trace).exists():
            print(f"{trace} does not exist")
            raise FileNotFoundError(f"{trace} does not exist")

    non_decision_input_dict = {
        f"q-{i + 1}": get_non_decision_inputs_for_q_compile(trace)
        for i, trace in enumerate(raw_traces)
    }
    non_decision_df = extract_non_decision_df(non_decision_input_dict)
    (
        graph_embeddings,
        non_decision_tabular_features,
        dt,
    ) = atomic_optimizer.extract_non_decision_embeddings_from_df(
        non_decision_df, ercilla=False
    )
    graph_embeddings = graph_embeddings.detach().cpu()
    df = non_decision_df.copy()
    ge_dim = graph_embeddings.shape[1]
    ge_cols = [f"ge_{i}" for i in range(ge_dim)]
    df[ge_cols] = graph_embeddings.numpy()
    ta = TypeAdvisor(q_type=q_type)
    df = df[ge_cols + ta.get_tabular_non_decision_columns()].copy()
    df["query_id"] = [f"{i + 1}-1" for i in range(22)]
    df.set_index("query_id", inplace=True, drop=True)
    os.makedirs(cache_header, exist_ok=True)
    ParquetHandler.save(df, cache_header, cache_file)
    print("non_decision_df saved...")
    return df, atomic_optimizer


def get_graph_embedding(
    ms: ModelServer,
    split_iterators: Dict[str, QueryPlanIterator],
    index_splits: Dict[str, np.ndarray],
    weights_header: str,
    name: str = "graph_np_dict.pkl",
) -> Dict[str, np.ndarray]:
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
    base_dir: Path,
    bm: str,
    q_type: QType,
    debug: bool,
    graph_choice: str,
    weights_path: Optional[str],
    if_df: bool = False,
    bm_target: Optional[str] = None,
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

    bm_target = bm_target or bm
    pw_model = PathWatcher(base_dir, bm, debug, extract_params)
    pw_data = PathWatcher(base_dir, bm_target, debug, extract_params)
    df = ParquetHandler.load(
        pw_data.cc_prefix, f"df_{ta.get_q_type_for_cache()}.parquet"
    )
    index_splits = PickleHandler.load(
        pw_data.cc_prefix, f"index_splits_{ta.get_q_type_for_cache()}.pkl"
    )
    if not isinstance(index_splits, Dict):
        raise TypeError(f"index_splits is not a dict: {index_splits}")
    objectives = ta.get_objectives()

    df_splits = {}
    df_splits_queries = {}
    if graph_choice == "none":
        for split, index in index_splits.items():
            df_split = df.loc[index].copy()
            df_split = df_split[ta.get_tabular_columns() + objectives]
            df_splits[split] = df_split
            df_split_queries = df.loc[index].copy()[["template", "qid"]]
            df_splits_queries[split] = df_split_queries
    elif graph_choice in ("avg", "tlstm", "qf", "gtn", "raal"):
        if graph_choice == "tlstm":
            model_sign = "tree_lstm"
        else:
            model_sign = f"graph_{graph_choice}"
        if (
            weights_path is None
            or not os.path.exists(weights_path)
            or len(weights_path.split("/")) != 7
        ):
            raise ValueError(f"weights_path is None: {weights_path}")
        header = "/".join(weights_path.split("/")[:4])
        model_params_path = (
            "/".join(weights_path.split("/")[:5]) + "/model_struct_params.json"
        )
        weights_header = "/".join(weights_path.split("/")[:6])
        ms = ModelServer.from_ckp_path(model_sign, model_params_path, weights_path)

        if bm_target == bm:
            split_iterators = PickleHandler.load(header, "split_iterators.pkl")
            if not isinstance(split_iterators, Dict):
                raise TypeError("split_iterators not found or not a desired type")
            graph_np_dict = get_graph_embedding(
                ms,
                split_iterators,
                index_splits,
                weights_header,
                name="graph_np_dict.pkl",
            )
        else:
            header = header.replace(
                f"{bm}_{pw_model.data_sign}", f"{bm_target}_{pw_data.data_sign}"
            )
            split_iterators = PickleHandler.load(header, "split_iterators.pkl")
            if not isinstance(split_iterators, Dict):
                raise TypeError("split_iterators not found or not a desired type")
            graph_np_dict = get_graph_embedding(
                ms,
                split_iterators,
                index_splits,
                weights_header,
                name=f"graph_np_dict_{bm_target}.pkl",
            )

        for split, index in index_splits.items():
            df_split = df.loc[index].copy()
            df_split = df_split[ta.get_tabular_columns() + ta.get_objectives()]
            df_split_queries = df.loc[index].copy()[["template", "qid"]]
            graph_np = graph_np_dict[split]
            ge_dim = graph_np.shape[1]
            ge_cols = [f"ge_{i}" for i in range(ge_dim)]
            df_split[ge_cols] = graph_np
            df_splits[split] = df_split
            df_splits_queries[split] = df_split_queries
    else:
        raise ValueError(f"Unknown graph choice: {graph_choice}")

    train_data = TabularDataset(df_splits["train"])
    val_data = TabularDataset(df_splits["val"])
    test_data = TabularDataset(df_splits["test"])
    data = (
        [df_splits["train"], df_splits["val"], df_splits["test"]]
        if if_df
        else [train_data, val_data, test_data]
    )
    data_queries = [
        df_splits_queries["train"],
        df_splits_queries["val"],
        df_splits_queries["test"],
    ]
    return {
        "data": data,
        "data_queries": data_queries,
        "ta": ta,
        "pw": pw_data,
        "objectives": objectives,
    }


def get_ag_pred_objs(
    base_dir: Path,
    bm: str,
    q_type: QType,
    debug: bool,
    graph_choice: str,
    split: str,
    ag_meta: Dict[str, str],
    force: bool,
    ag_model: Dict[str, str],
    bm_target: Optional[str] = None,
    xfer_gtn_only: bool = False,
    tpl_plus: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, float, float, Dict]:
    weights_head = (
        os.path.dirname(ag_meta["graph_weights_path"])
        if graph_choice != "none"
        else f"cache_and_ckp/{bm}_{get_data_sign(bm, debug)}/{q_type}/none"
    )
    ag_name_splits = ag_meta["ag_full_name"].split("_")
    ag_sign = "_".join(ag_name_splits[:1] + ag_name_splits[2:])
    ag_model_short = "_".join(f"{k.split('_')[0]}:{v}" for k, v in ag_model.items())
    device = "gpu" if th.cuda.is_available() else "cpu"
    bm_target = bm_target or bm
    if bm_target != bm and not xfer_gtn_only:
        cache_name = (
            f"{split}_{ag_sign}_{ag_model_short}_objs_and_metrics_"
            f"for_{bm_target}_{device}.pkl"
        )
    else:
        cache_name = f"{split}_{ag_sign}_{ag_model_short}_objs_and_metrics_{device}.pkl"

    if not force and os.path.exists(f"{weights_head}/{cache_name}"):
        print(f"found {cache_name}")
        cache = PickleHandler.load(weights_head, cache_name)
        if not isinstance(cache, Dict):
            raise TypeError(f"mlp_cache is not a dict: {cache}")
        (
            objs_true,
            objs_pred,
        ) = (
            cache["objs_true"],
            cache["objs_pred"],
        )
        dt_s, throughput = cache["dt_s"], cache["throughput"]
        metrics = cache["metrics"]
        return objs_true, objs_pred, dt_s, throughput, metrics

    print(f"not found {weights_head}/{cache_name}, generating...")
    ag_data = get_ag_data(
        base_dir,
        bm,
        q_type,
        debug,
        graph_choice,
        ag_meta["graph_weights_path"] if graph_choice != "none" else None,
        bm_target=bm_target,
    )
    ta, objectives = ag_data["ta"], ag_data["objectives"]
    ag_path = ag_meta["ag_path"]

    if bm_target != bm and xfer_gtn_only:
        ag_path += f"_{bm_target}"

    if tpl_plus:
        ag_path += "_plus_tpl"
        data_dict = {
            sp: da.join(daq[["template"]].astype("str"))
            for sp, da, daq in zip(
                ["train", "val", "test"], ag_data["data"], ag_data["data_queries"]
            )
        }
        data = data_dict[split]
    else:
        data_dict = {
            sp: da for sp, da in zip(["train", "val", "test"], ag_data["data"])
        }
        data = data_dict[split]

    objectives = ta.get_ag_objectives()
    dt_ns = 0
    y_pred_list = []
    transformed_data = None
    for obj in objectives:
        predictor = TabularPredictor.load(f"{ag_path}/Predictor_{obj}")
        predictor.persist()
        if transformed_data is None:
            transformed_data = predictor.transform_features(
                data.drop(columns=objectives)
            )
        # print(predictor.model_names())
        start_time_ns = time.perf_counter_ns()
        y_pred = predictor.predict(
            transformed_data,
            model=ag_model[obj],
            as_pandas=False,
            transform_features=False,
        )
        y_pred = calibrate_negative_predictions(y_pred, bm, obj, q_type)
        y_pred_list.append(y_pred)
        end_time_ns = time.perf_counter_ns()
        dt_ns += end_time_ns - start_time_ns
        predictor.unpersist()
    objs_true = data[objectives]
    objs_pred = pd.DataFrame(np.vstack(y_pred_list).T, columns=objectives)
    dt_s = dt_ns / 1e9
    throughput = len(data) / 1e3 / dt_s

    metrics = get_metric_stats(objectives, objs_true, objs_pred)
    PickleHandler.save(
        {
            "objs_true": objs_true,
            "objs_pred": objs_pred,
            "dt_s": dt_s,
            "throughput": throughput,
            "metrics": metrics,
        },
        weights_head,
        cache_name,
        overwrite=force,
    )

    return objs_true, objs_pred, dt_s, throughput, metrics


def get_metric_stats(
    objectives: List[str], objs_true: pd.DataFrame, objs_pred: pd.DataFrame
) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    if len(objs_true) == 0:
        return metrics
    for obj_ind, obj in enumerate(objectives):
        metrics[obj] = {}
        y = objs_true[obj].to_numpy()
        y_pred = objs_pred[obj].to_numpy()
        metrics[obj]["wmape"] = local_wmape(y, y_pred)
        metrics[obj]["p50_err"] = local_p50_err(y, y_pred)
        metrics[obj]["p90_err"] = local_p90_err(y, y_pred)
        metrics[obj]["p50_wape"] = local_p50_wape(y, y_pred)
        metrics[obj]["p90_wape"] = local_p90_wape(y, y_pred)
        metrics[obj]["corr"] = float(np.corrcoef(y, y_pred)[0, 1])
    return metrics


def get_mlp_pred_objs(
    bm: str,
    q_type: QType,
    debug: bool,
    graph_choice: str,
    weights_path: str,
    split: str,
    force: bool,
    bm_target: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, float, float, Dict]:
    ta = TypeAdvisor(q_type=q_type)
    bm_target = bm_target or bm

    weights_head = os.path.dirname(weights_path)
    device = "gpu" if th.cuda.is_available() else "cpu"
    cache_name = f"{split}_mlp_objs_{device}.pkl"
    if not force and os.path.exists(f"{weights_head}/{cache_name}"):
        # print(f"found {weights_head}/{cache_name}")
        cache = PickleHandler.load(weights_head, cache_name)
        if not isinstance(cache, Dict):
            raise TypeError(f"mlp_cache is not a dict: {cache}")
        (
            objs_true,
            objs_pred,
        ) = (
            cache["objs_true"],
            cache["objs_pred"],
        )
        dt_s, throughput = cache["dt_s"], cache["throughput"]
        if "metrics" not in cache:
            metrics: Dict[str, Any] = {}
            for obj_ind, obj in enumerate(ta.get_objectives()):
                metrics[obj] = {}
                y = objs_true[obj].values
                y_pred = objs_pred[obj].values
                metrics[obj]["wmape"] = local_wmape(y, y_pred)
                metrics[obj]["p50_err"] = local_p50_err(y, y_pred)
                metrics[obj]["p90_err"] = local_p90_err(y, y_pred)
                metrics[obj]["p50_wape"] = local_p50_wape(y, y_pred)
                metrics[obj]["p90_wape"] = local_p90_wape(y, y_pred)
                metrics[obj]["corr"] = float(np.corrcoef(y, y_pred)[0, 1])
        else:
            metrics = cache["metrics"]
        return objs_true, objs_pred, dt_s, throughput, metrics

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
    pw_model = PathWatcher(Path(__file__).parent, bm, debug, extract_params)
    pw_data = PathWatcher(Path(__file__).parent, bm_target, debug, extract_params)
    df = ParquetHandler.load(
        pw_data.cc_prefix, f"df_{ta.get_q_type_for_cache()}.parquet"
    )
    index_splits = PickleHandler.load(
        pw_data.cc_prefix, f"index_splits_{ta.get_q_type_for_cache()}.pkl"
    )
    if not isinstance(index_splits, Dict):
        raise TypeError(f"index_splits is not a dict: {index_splits}")
    objectives = ta.get_objectives()
    if graph_choice not in ("avg", "gtn"):
        raise ValueError(f"graph_choice {graph_choice} not supported")
    if (
        weights_path is None
        or not os.path.exists(weights_path)
        or len(weights_path.split("/")) != 7
    ):
        raise ValueError("weights_path is None")

    model_sign = f"graph_{graph_choice}"
    header = "/".join(weights_path.split("/")[:4])
    model_params_path = (
        "/".join(weights_path.split("/")[:5]) + "/model_struct_params.json"
    )
    weights_header = "/".join(weights_path.split("/")[:6])
    ms = ModelServer.from_ckp_path(model_sign, model_params_path, weights_path)
    if bm_target == bm:
        split_iterators = PickleHandler.load(header, "split_iterators.pkl")
        if not isinstance(split_iterators, Dict):
            raise TypeError("split_iterators not found or not a desired type")

        graph_np_dict = get_graph_embedding(
            ms, split_iterators, index_splits, weights_header, name="graph_np_dict.pkl"
        )
    else:
        header = header.replace(
            f"{bm}_{pw_model.data_sign}", f"{bm_target}_{pw_data.data_sign}"
        )
        split_iterators = PickleHandler.load(header, "split_iterators.pkl")
        if not isinstance(split_iterators, Dict):
            raise TypeError("split_iterators not found or not a desired type")
        split_iterators = {"test": split_iterators["test"]}

        graph_np_dict = get_graph_embedding(
            ms,
            split_iterators,
            index_splits,
            weights_header,
            name=f"graph_np_dict_{bm_target}.pkl",
        )
    index = index_splits[split]
    iterator = split_iterators[split]
    if not isinstance(iterator, QueryPlanIterator):
        raise TypeError(f"iterator is not a QueryPlanIterator: {iterator}")

    if split == "train":
        bs = 32768
        dataloader = iterator.get_dataloader(
            batch_size=bs, shuffle=False, num_workers=32
        )
        ge = graph_np_dict[split]

        all_preds = []
        ge_tensor = th.tensor(ge, dtype=th.float32, device=device)
        start_index = 0
        start_time_ns = time.perf_counter_ns()
        for input, _ in dataloader:
            tabular_features = input.features
            with th.no_grad():
                objs_pred = (
                    ms.model.regressor(
                        ge_tensor[start_index : start_index + bs],
                        tabular_features.to(device),
                    )
                    .detach()
                    .cpu()
                )
            all_preds.append(objs_pred)
            start_index += bs
        objs_pred = th.cat(all_preds, dim=0)
        end_time_ns = time.perf_counter_ns()
    else:
        dataloader = iterator.get_dataloader(
            batch_size=len(index), shuffle=False, num_workers=32
        )
        input, _ = next(iter(dataloader))
        tabular_features = input.features
        ge = graph_np_dict[split]

        start_time_ns = time.perf_counter_ns()
        with th.no_grad():
            objs_pred = (
                ms.model.regressor(
                    th.tensor(ge, dtype=th.float32, device=device),
                    tabular_features.to(device),
                )
                .detach()
                .cpu()
            )
        end_time_ns = time.perf_counter_ns()

    dt_ns = end_time_ns - start_time_ns
    dt_s = dt_ns / 1e9
    throughput = len(index) / 1e3 / dt_s
    objs_true = df.loc[index][objectives]
    objs_pred = pd.DataFrame(objs_pred.numpy(), columns=objectives)

    metrics = {}
    for obj_ind, obj in enumerate(objectives):
        metrics[obj] = {}
        y = objs_true[obj].values
        y_pred = objs_pred[obj].values
        metrics[obj]["wmape"] = local_wmape(y, y_pred)
        metrics[obj]["p50_err"] = local_p50_err(y, y_pred)
        metrics[obj]["p90_err"] = local_p90_err(y, y_pred)
        metrics[obj]["p50_wape"] = local_p50_wape(y, y_pred)
        metrics[obj]["p90_wape"] = local_p90_wape(y, y_pred)
        metrics[obj]["corr"] = float(np.corrcoef(y, y_pred)[0, 1])

    PickleHandler.save(
        {
            "objs_true": objs_true,
            "objs_pred": objs_pred,
            "dt_s": dt_s,
            "throughput": throughput,
            "metrics": metrics,
        },
        weights_head,
        cache_name,
        overwrite=force,
    )

    return objs_true, objs_pred, dt_s, throughput, metrics


def get_alias(q_type: QType) -> str:
    if q_type == "q_compile":
        return "Q_compile"
    if q_type == "q_all":
        return "Q_runtime"
    if q_type == "qs_lqp_compile":
        return "subQ_compile"
    if q_type == "qs_lqp_runtime":
        return "subQ_runtime"
    if q_type == "qs_pqp_runtime":
        return "QS_runtime"
    raise ValueError(f"Unknown q_type: {q_type}")


def get_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> List[float]:
    wmape = local_wmape(y_true, y_pred)
    p50_wape = local_p50_wape(y_true, y_pred)
    p90_wape = local_p90_wape(y_true, y_pred)
    # p90_err = local_p90_err(y_true, y_pred)
    corr = float(np.corrcoef(y_true, y_pred)[0, 1])
    res = [wmape, p50_wape, p90_wape, corr]
    return res


def summarize_metrics(
    q_type: QType,
    graph_choice: str,
    objs_true: pd.DataFrame,
    objs_pred: pd.DataFrame,
    throughput_k_s: float,
    model_suffix: str,
) -> list[Union[float, str]]:
    print(
        f"q_type: {q_type}, graph_choice: {graph_choice}, model_suffix: {model_suffix}"
    )
    graph_choice = f"{graph_choice}+{model_suffix}"
    if q_type.startswith("q_"):
        lat_true = objs_true["latency_s"].to_numpy()
        lat_pred = objs_pred["latency_s"].to_numpy()
        io_true = objs_true["io_mb"].to_numpy()
        io_pred = objs_pred["io_mb"].to_numpy()
        m_lat = get_metrics(lat_true, lat_pred)
        m_io = get_metrics(io_true, io_pred)
        return (
            [get_alias(q_type), graph_choice]
            + m_lat
            + [np.nan] * len(m_lat)
            + m_io
            + [throughput_k_s]
        )

    elif q_type.startswith("qs_"):
        ana_lat_true = objs_true["ana_latency_s"].to_numpy()
        ana_lat_pred = objs_pred["ana_latency_s"].to_numpy()
        io_true = objs_true["io_mb"].to_numpy()
        io_pred = objs_pred["io_mb"].to_numpy()
        m_ana_lat = get_metrics(ana_lat_true, ana_lat_pred)
        m_io = get_metrics(io_true, io_pred)
        return (
            [get_alias(q_type), graph_choice]
            + [np.nan] * len(m_ana_lat)
            + m_ana_lat
            + m_io
            + [throughput_k_s]
        )

    else:
        raise ValueError(f"Unknown q_type: {q_type}")


def display_ape(f: Optional[Union[str, float, int]]) -> str:
    if isinstance(f, str):
        return f
    if f is None:
        return "-"
    if f < 0.001:
        return f"{f:.0e}"
    if f < 10:
        return f"{round(f, 3):.3f}"
    else:  # f >= 10
        return f"{round(f, 2):.2f}"


def display_xput(f: Union[str, float, int]) -> str:
    if isinstance(f, str):
        return f
    if f < 1:
        return f"{round(f, 2):.2f}"
    if f < 10:
        return f"{round(f, 1):.1f}"
    else:  # f >= 10
        return f"{round(f):.0f}"
