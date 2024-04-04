import os
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

from udao_spark.utils.evaluation import display_ape, display_xput
from udao_spark.utils.params import get_base_parser
from udao_trace.utils import JsonHandler, PickleHandler


def get_parser() -> ArgumentParser:
    parser = get_base_parser()
    # fmt: off
    parser.add_argument("--hp_choice", type=str, default="tuned-0215",
                        choices=["tuned-0215"])
    # fmt: on
    return parser


if __name__ == "__main__":
    params = get_parser().parse_args()
    if params.q_type != "q_compile":
        raise ValueError(f"Diagnosing {params.q_type} is not our focus.")
    if params.hp_choice != "tuned-0215":
        raise ValueError(f"hp_choice {params.hp_choice} is not supported.")

    bm = params.benchmark
    base_dir = Path(__file__).parent
    res_pkl = JsonHandler.load_json("assets/res_pkl.json")
    objs = ["latency_s", "io_mb"]
    metric_list = ["wmape", "p50_wape", "p90_wape", "corr"]
    print(f"modeling results for {bm}")
    data = []

    columns = [
        "model",
        "lat_wmape",
        "lat_p50",
        "lat_p90",
        "lat_corr",
        "io_wmape",
        "io_p50",
        "io_p90",
        "io_corr",
        "throughput(K/s)",
    ]

    for model_name in [
        "qppnet",
        "tlstm",
        "qf",
        "raal",
        "avg",
        "gtn",
        "em(fast)",
        "em(best)",
    ]:
        row = [model_name]
        path = res_pkl[bm][model_name]
        cache = PickleHandler.load(os.path.dirname(path), os.path.basename(path))
        if not isinstance(cache, dict):
            raise ValueError(f"invalid cache for {model_name}")
        for obj in objs:
            row += [cache["metrics"][obj][m] for m in metric_list]
        if model_name.startswith("em"):
            throughput = cache["throughput"]
        else:
            data_prepare_ms = cache["time_eval"]["data_prepare_ms"]
            pred_ms = cache["time_eval"]["pred_ms"]
            throughput = cache["obj_df"].shape[0] / pred_ms
        row.append(throughput)
        print("\t".join(map(str, row)))
        data.append(row)

    df = pd.DataFrame(data, columns=columns)
    ape_cols = [c for c in df.columns if "throughput" not in c and c != "model_name"]
    xput_cols = [c for c in df.columns if "throughput" in c]
    df[ape_cols] = df[ape_cols].applymap(display_ape)
    df[xput_cols] = df[xput_cols].applymap(display_xput)
    print(df[df.columns[1:]].to_latex(index=False))
