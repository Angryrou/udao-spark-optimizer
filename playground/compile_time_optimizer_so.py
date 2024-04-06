import os
import time
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch as th

from udao_spark.data.extractors.injection_extractor import (
    get_non_decision_inputs_for_q_compile,
)
from udao_spark.data.utils import get_lhs_confs
from udao_spark.optimizer.atomic_optimizer import AtomicOptimizer
from udao_spark.optimizer.utils import get_ag_meta
from udao_spark.utils.logging import logger
from udao_spark.utils.params import get_ag_parameters
from udao_trace.configuration import SparkConf
from udao_trace.utils import BenchmarkType, JsonHandler
from udao_trace.workload import Benchmark


def get_params() -> ArgumentParser:
    parser = get_ag_parameters()
    # fmt: off
    parser.add_argument("--demo", action="store_true",
                        help="demo mode")
    parser.add_argument("--ag_model_q_latency", type=str, default=None,
                        help="specific model name for AG for latency",)
    parser.add_argument("--ag_model_q_io", type=str, default=None,
                        help="specific model name for AG for IO",)
    parser.add_argument("--n_conf_samples", type=int, default=10000,
                        help="number of configurations to sample")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose mode")
    parser.add_argument("--ensemble", action="store_true",
                        help="Enable verbose mode")
    # fmt: on
    return parser


if __name__ == "__main__":
    params = get_params().parse_args()
    if params.q_type != "q_compile":
        raise ValueError(f"Diagnosing {params.q_type} is not our focus.")
    if params.hp_choice != "tuned-0215":
        raise ValueError(f"hp_choice {params.hp_choice} is not supported.")
    if params.ensemble and params.graph_choice != "gtn":
        raise ValueError(f"ensembled model only works with {params.graph_choice}.")

    # theta includes 19 decision variables
    decision_variables = (
        ["k1", "k2", "k3", "k4", "k5", "k6", "k7", "k8"]
        + ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9"]
        + ["s10", "s11"]
    )
    base_dir = Path(__file__).parent
    bm, q_type = params.benchmark, params.q_type
    use_ag = params.ensemble

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

    hp_choice, graph_choice = params.hp_choice, params.graph_choice
    ag_sign = params.ag_sign
    il, bs, tl = params.infer_limit, params.infer_limit_batch_size, params.ag_time_limit
    ag_meta = get_ag_meta(bm, hp_choice, graph_choice, q_type, ag_sign, il, bs, tl)
    ag_model = {
        "latency_s": params.ag_model_q_latency,
        "io_mb": params.ag_model_q_io,
    }
    # use functions in HierarchicalOptimizer to extract the non-decision inputs
    atomic_optimizer = AtomicOptimizer(
        bm=bm,
        model_sign=ag_meta["model_sign"],
        graph_model_params_path=ag_meta["model_params_path"],
        graph_weights_path=ag_meta["graph_weights_path"],
        q_type=q_type,
        data_processor_path=ag_meta["data_processor_path"],
        spark_conf=spark_conf,
        decision_variables=decision_variables,
        ag_path=ag_meta["ag_path"],
        clf_json_path=None
        if params.disable_failure_clf
        else str(base_dir / f"assets/{bm}_valid_clf_meta.json"),
        clf_recall_xhold=params.clf_recall_xhold,
        verbose=False,
    )

    todo_confs: Dict[str, Dict] = {}
    target_confs: Dict[str, Dict] = {}
    total_monitor = {}
    n_samples = params.n_conf_samples
    seed = params.seed
    sampled_theta = get_lhs_confs(
        atomic_optimizer.sc, n_samples, seed=seed, normalize=not use_ag
    ).values

    for template, trace in zip(benchmark.templates, raw_traces):
        logger.info(f"Processing {trace}")
        query_id = f"{template}-1"
        start = time.perf_counter_ns()
        non_decision_input = get_non_decision_inputs_for_q_compile(trace)
        non_decision_df = atomic_optimizer.extract_non_decision_df(non_decision_input)
        (
            graph_embeddings,
            non_decision_tabular_features,
            time_dict,
        ) = atomic_optimizer.extract_non_decision_embeddings_from_df(
            non_decision_df, use_ag, ercilla=False
        )
        regr_start = time.perf_counter_ns()
        if use_ag:
            graph_embeddings = graph_embeddings.detach().cpu()
            objs_dict = atomic_optimizer.get_objective_values_ag(
                template,
                graph_embeddings.tile(n_samples, 1).numpy(),
                pd.DataFrame(
                    np.tile(non_decision_df.values, (n_samples, 1)),
                    columns=non_decision_df.columns,
                ),
                sampled_theta,
                ag_model,
            )
        else:
            if non_decision_tabular_features is None:
                raise ValueError(
                    "non_decision_tabular_features is required for MLP inference"
                )
            objs_dict = atomic_optimizer.get_objective_values_mlp(
                graph_embeddings.tile(n_samples, 1),
                non_decision_tabular_features.tile(n_samples, 1),
                th.tensor(sampled_theta, dtype=atomic_optimizer.dtype),
            )
        lat, cost = atomic_optimizer.get_latencies_and_objectives(objs_dict)
        end = time.perf_counter_ns()
        time_dict["regr_ms"] = (end - regr_start) / 1e6
        time_dict["total_ms"] = (end - start) / 1e6
        time_dict["total_confs"] = n_samples
        total_monitor[query_id] = time_dict
        ind = np.lexsort((cost, lat))[0]
        print(lat.sum())
        po_conf = atomic_optimizer.construct_po_confs(
            sampled_theta[ind : ind + 1], use_ag
        )
        todo_confs[query_id] = [",".join(p) for p in po_conf]  # type: ignore

    total_ms = sum([v["total_ms"] for v in total_monitor.values()])
    total_confs = sum([v["total_confs"] for v in total_monitor.values()])
    print(
        f"Model: {graph_choice}, total_ms: {total_ms}, total_confs: {total_confs}, "
        f"xput(K/s): {total_confs / total_ms}"
    )
    total_monitor["agg"] = {
        "total_ms": total_ms,
        "total_confs": total_confs,
        "xput": total_confs / total_ms,
    }
    device = "gpu" if th.cuda.is_available() else "cpu"
    if not use_ag:
        suffix = f"{n_samples}_{graph_choice}_{device}"
    else:
        ag_model_short = "_".join(f"{k.split('_')[0]}:{v}" for k, v in ag_model.items())
        suffix = f"{n_samples}_{graph_choice}_em({ag_model_short})_{device}"

    torun_file = f"compile_time_output/{bm}100/lhs-so/uco-run_confs_{suffix}.json"
    runtime_file = f"compile_time_output/{bm}100/lhs-so/uco-runtime_{suffix}.json"

    os.makedirs(os.path.dirname(torun_file), exist_ok=True)
    JsonHandler.dump_to_file(todo_confs, torun_file, indent=2)
    JsonHandler.dump_to_file(total_monitor, runtime_file, indent=2)
