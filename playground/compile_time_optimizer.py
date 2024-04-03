"""
Support compile-time configuration recommendation with the following features:
- EM models and MLP models at the query level
- Different objective preferences
- Different model uncertainty preferences (TODO)
"""
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict

import numpy as np

from udao_spark.data.extractors.injection_extractor import (
    get_non_decision_inputs_for_q_compile,
)
from udao_spark.optimizer.atomic_optimizer import AtomicOptimizer
from udao_spark.optimizer.utils import get_ag_meta, weighted_utopia_nearest
from udao_spark.utils.logging import logger
from udao_spark.utils.monitor import UdaoMonitor
from udao_spark.utils.params import QType, get_ag_parameters
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
    # fmt: on
    return parser


if __name__ == "__main__":
    R_Q: QType = "q_compile"
    params = get_params().parse_args()
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
    bm, q_type = params.benchmark, params.q_type
    hp_choice, graph_choice = params.hp_choice, params.graph_choice
    ag_sign = params.ag_sign
    il, bs, tl = params.infer_limit, params.infer_limit_batch_size, params.ag_time_limit
    ag_meta = get_ag_meta(bm, hp_choice, graph_choice, q_type, ag_sign, il, bs, tl)
    ag_full_name = ag_meta["ag_full_name"]

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
        decision_variables=decision_variables,
        ag_path=ag_meta["ag_path"],
        clf_json_path=None
        if params.disable_failure_clf
        else str(base_dir / f"assets/{bm}_valid_clf_meta.json"),
        clf_recall_xhold=params.clf_recall_xhold,
        verbose=False,
    )

    torun_dict: Dict[str, Dict] = {}
    wun_weights_pairs = {
        0: (0.0, 1.0),
        1: (0.1, 0.9),
        5: (0.5, 0.5),
        9: (0.9, 0.1),
        10: (1.0, 0.0),
    }
    target_confs: Dict[str, Dict] = {}
    total_monitor = {}
    n_samples = params.n_conf_samples
    for template, trace in zip(benchmark.templates, raw_traces):
        logger.info(f"Processing {trace}")
        query_id = f"{template}-1"
        non_decision_input = get_non_decision_inputs_for_q_compile(trace)
        monitor = UdaoMonitor()
        po_objs, po_confs = atomic_optimizer.solve(
            template=template,
            non_decision_input=non_decision_input,
            seed=params.seed,
            ag_model={
                "latency_s": params.ag_model_q_latency,
                "io_mb": params.ag_model_q_io,
            },
            sample_mode="lhs",
            n_samples=n_samples,
            monitor=monitor,
        )
        total_monitor[query_id] = monitor.to_dict()
        if po_objs is None or po_confs is None:
            logger.warning(f"Failed to solve {template}")
            continue
        target_confs[query_id] = {}
        for k, wun_weights in wun_weights_pairs.items():
            reco_obj, reco_conf = weighted_utopia_nearest(
                pareto_objs=po_objs,
                pareto_confs=po_confs,
                weights=np.array(wun_weights),
            )
            target_confs[query_id][k] = ",".join(reco_conf)

    todo_confs = {
        query_id: np.unique([c for c in confs_dict.values()]).tolist()
        for query_id, confs_dict in target_confs.items()
    }
    torun_file = f"compile_time_output/{bm}100/lhs/run_confs_{n_samples}.json"
    toanalyze_file = f"compile_time_output/{bm}100/lhs/full_confs_{n_samples}.json"
    runtime_file = f"compile_time_output/{bm}100/lhs/runtime_{n_samples}.json"
    os.makedirs(os.path.dirname(torun_file), exist_ok=True)
    JsonHandler.dump_to_file(target_confs, toanalyze_file, indent=2)
    JsonHandler.dump_to_file(todo_confs, torun_file, indent=2)
    JsonHandler.dump_to_file(total_monitor, runtime_file, indent=2)
