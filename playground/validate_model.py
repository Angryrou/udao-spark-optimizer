import os
from argparse import Namespace
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from udao_spark.data.extractors.injection_extractor import (
    get_non_decision_inputs_for_q_compile,
)
from udao_spark.model.model_server import AGServer
from udao_spark.optimizer.atomic_optimizer import AtomicOptimizer
from udao_spark.optimizer.utils import get_ag_meta
from udao_spark.utils.collaborators import TypeAdvisor
from udao_spark.utils.evaluation import extract_non_decision_df
from udao_spark.utils.params import get_ag_parameters
from udao_trace.configuration import SparkConf
from udao_trace.utils import BenchmarkType, ParquetHandler
from udao_trace.workload import Benchmark


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
    fold = params.fold
    ag_meta = get_ag_meta(
        bm, hp_choice, graph_choice, q_type, ag_sign, il, bs, tl, fold
    )
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


if __name__ == "__main__":
    params = get_ag_parameters().parse_args()
    if params.q_type != "q_compile":
        raise ValueError(f"Diagnosing {params.q_type} is not our focus.")
    if params.hp_choice != "tuned-0215":
        raise ValueError(f"hp_choice {params.hp_choice} is not supported.")
    if params.graph_choice != "gtn":
        raise ValueError(f"graph_choice {params.graph_choice} is not supported.")

    # theta includes 19 decision variables
    decision_variables = (
        ["k1", "k2", "k3", "k4", "k5", "k6", "k7", "k8"]
        + ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9"]
        + ["s10", "s11"]
    )
    base_dir = Path(__file__).parent
    df, optimizer = get_non_decision_inputs(
        base_dir=base_dir,
        params=params,
        decision_vars=decision_variables,
        cache_file="non_decision_df_03-07.parquet",
    )
    ag_server = optimizer.ag_ms
    # example of setting different theta for TPCH q1-1
    target_query = "1-1"

    # 1. get non decision inputs from input_dict
    target_df = df.loc[[target_query]]
    if target_df.empty:
        raise ValueError(f"target query {target_query} not found")

    # 2. prepare theta for the target query
    base_dir = Path(__file__).parent
    spark_conf = SparkConf(str(base_dir / "assets/spark_configuration_aqe_on.json"))
    # note that the knob values are SCALED to integer from the raw traces
    # more details are in assets/spark_configuration_aqe_on.json
    theta_anchor = [1, 1, 16, 4, 5, 1, 1, 75, 5, 6, 32, 32, 50, 4, 80, 4, 4, 35, 6]
    theta_lower_bounds = np.array(spark_conf.knob_min)
    theta_upper_bounds = np.array(spark_conf.knob_max)
    if np.any(np.array(theta_anchor) < theta_lower_bounds) or np.any(
        np.array(theta_anchor) > theta_upper_bounds
    ):
        raise Exception("invalid values in theta_anchor")

    target_df[decision_variables] = theta_anchor
    # you can play with the following three knobs
    # k1 (#cores/exec),
    # k2 (mem-G/cores) => mem-G/exec = k1 * k2
    # k3 (#exec) and fix other knobs.
    # total cores = k1 * k3
    # total memory = k1 * k2 * k3
    # e.g.,
    k1k2k3_list = [
        [1, 1, 4],  # 1*4=4 cores, 1*1*4=4GB, 4 exec (minimum resources setting)
        [1, 1, 16],  # 1*16=16 core, 1*1*16=16GB, 16 exec (default resources setting)
        [5, 4, 16],  # 5*16=80 cores, 5*4*16=320GB, 16 exec (maximum resources setting)
    ]
    target_df = pd.DataFrame(
        np.tile(target_df.values, (len(k1k2k3_list), 1)), columns=target_df.columns
    )
    target_df[["k1", "k2", "k3"]] = k1k2k3_list
    if np.any(target_df[decision_variables] < theta_lower_bounds) or np.any(
        target_df[decision_variables] > theta_upper_bounds
    ):
        raise Exception("invalid decision variables in target_df")

    # 3. get the latency predictor
    lat_predictor = ag_server.predictors["latency_s"]
    lat_predictor_path = lat_predictor.path
    print(f"model lat_predictor_path: {lat_predictor_path}")

    # 4. predict the latency
    lat_pred = lat_predictor.predict(target_df, model="WeightedEnsemble_L2")

    print(lat_pred)
