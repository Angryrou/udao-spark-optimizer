import os
from argparse import Namespace
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from udao_spark.data.extractors.injection_extractor import (
    get_non_decision_inputs_for_q_compile,
)
from udao_spark.model.model_server import AGServer
from udao_spark.optimizer.atomic_optimizer import AtomicOptimizer
from udao_spark.optimizer.utils import get_ag_meta
from udao_spark.utils.collaborators import TypeAdvisor
from udao_trace.configuration import SparkConf
from udao_trace.utils import BenchmarkType, ParquetHandler
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
    base_dir: Path, params: Namespace, decision_vars: List[str]
) -> Tuple[pd.DataFrame, AGServer]:
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
    cache_file = "non_decision_df.parquet"

    try:
        df = ParquetHandler.load(cache_header, cache_file)
        ag_ms = AGServer.from_ckp_path(
            model_sign=ag_meta["model_sign"],
            graph_model_params_path=ag_meta["model_params_path"],
            graph_weights_path=ag_meta["graph_weights_path"],
            q_type=q_type,
            ag_path=ag_meta["ag_path"],
            clf_json_path=str(base_dir / f"assets/{bm}_valid_clf_meta.json"),
            clf_recall_xhold=params.clf_recall_xhold,
        )
        print("found cached non_decision_df...")
        return df, ag_ms
    except FileNotFoundError:
        print("no cached non_decision_df found, generating...")
    except Exception as e:
        print(f"error loading cached non_decision_df: {e}")
        raise e

    # prepare the traces
    spark_conf = SparkConf(str(base_dir / "assets/spark_configuration_aqe_on.json"))
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

    non_decision_input_dict = {
        f"q-{i + 1}": get_non_decision_inputs_for_q_compile(trace)
        for i, trace in enumerate(raw_traces)
    }
    non_decision_df = extract_non_decision_df(non_decision_input_dict)
    (
        graph_embeddings,
        non_decision_tabular_features,
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
    df["query_name"] = [f"q1-{i + 1}" for i in range(22)]
    df.set_index("query_name", inplace=True, drop=True)
    os.makedirs(cache_header, exist_ok=True)
    ParquetHandler.save(df, cache_header, cache_file)
    print("non_decision_df saved...")
    return df, atomic_optimizer.ag_ms
