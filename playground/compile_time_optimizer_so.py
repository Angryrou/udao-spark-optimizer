import os
import time
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch as th

from udao_spark.data.extractors.injection_extractor import (
    get_non_decision_inputs_for_q_compile,
)
from udao_spark.data.utils import get_lhs_confs, wrap_to_df
from udao_spark.optimizer.setup import q_compile_setup
from udao_spark.utils.logging import logger
from udao_spark.utils.params import get_ag_parameters
from udao_trace.utils import JsonHandler


def get_params() -> ArgumentParser:
    parser = get_ag_parameters()
    # fmt: off
    parser.add_argument("--ag_model_q_latency", type=str, default=None,
                        help="specific model name for AG for latency",)
    parser.add_argument("--ag_model_q_io", type=str, default=None,
                        help="specific model name for AG for IO",)
    parser.add_argument("--n_conf_samples", type=int, default=10000,
                        help="number of configurations to sample")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose mode")
    parser.add_argument("--ensemble", action="store_true",
                        help="Enable ensemble mode")
    parser.add_argument("--default-rate", type=float, default=None)
    # fmt: on
    return parser


if __name__ == "__main__":
    params = get_params().parse_args()
    base_dir = Path(__file__).parent

    meta = q_compile_setup(base_dir, params)
    atomic_optimizer = meta["atomic_optimizer"]
    templates = meta["templates"]
    raw_traces = meta["raw_traces"]
    ag_model = meta["ag_model"]

    device = "gpu" if th.cuda.is_available() else "cpu"
    use_ag = params.ensemble
    graph_choice = params.graph_choice
    bm = params.benchmark

    todo_confs: Dict[str, List] = {}
    target_confs: Dict[str, List] = {}
    target_objs: Dict[str, Dict] = {}
    total_monitor = {}
    n_samples = params.n_conf_samples
    seed = params.seed
    sampled_theta = get_lhs_confs(
        atomic_optimizer.sc, n_samples, seed=seed, normalize=not use_ag
    ).values

    default_rate = params.default_rate
    df_default: Optional[pd.DataFrame] = None
    if default_rate is not None:
        d = JsonHandler.load_json(
            str(base_dir / f"assets/default_evaluations/{bm}.json")
        )
        df_default = wrap_to_df(d["agg_stats"])

    for template, trace in zip(templates, raw_traces):
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
            non_decision_df, use_ag, ercilla=True, graph_choice=graph_choice
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

        if df_default is not None:
            cost_constraints = df_default.loc[query_id, "cost_w_io_mu"] * default_rate
            valid_inds = np.where(cost <= cost_constraints)[0]
            lat = lat[valid_inds]
            cost = cost[valid_inds]
            filtered_theta = sampled_theta[valid_inds]
            logger.info(f"got {len(valid_inds)}/{n_samples} samples after filtering")
        else:
            filtered_theta = sampled_theta

        if len(lat) == 0:
            todo_confs[query_id] = []
            target_objs[query_id] = {}
            logger.warning(f"Failed to solve {template} !!!!!")
            continue

        ind = np.lexsort((cost, lat))[0]
        print(lat.sum())
        po_conf = atomic_optimizer.construct_po_confs(
            filtered_theta[ind : ind + 1], use_ag
        )
        todo_confs[query_id] = [",".join(p) for p in po_conf]
        target_objs[query_id] = {
            "latency_s_hat": float(lat[ind]),
            "cost_hat": float(cost[ind]),
        }

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

    if df_default is None:
        prefix = "uco"
    else:
        prefix = "co"
        suffix += f"_dr{default_rate}"
    torun_file = f"compile_time_output/{bm}100/lhs-so/{prefix}-run_confs_{suffix}.json"
    runtime_file = f"compile_time_output/{bm}100/lhs-so/{prefix}-runtime_{suffix}.json"
    target_objs_file = f"compile_time_output/{bm}100/lhs-so/{prefix}-objs_{suffix}.json"

    os.makedirs(os.path.dirname(torun_file), exist_ok=True)
    JsonHandler.dump_to_file(todo_confs, torun_file, indent=2)
    JsonHandler.dump_to_file(total_monitor, runtime_file, indent=2)
    JsonHandler.dump_to_file(target_objs, target_objs_file, indent=2)
