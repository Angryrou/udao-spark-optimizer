"""
To build to DataFrame from the raw traces in one path

DF1 (q.csv)
appid, tid, qid, lqp_id, lqp(json), im-1, im-2, pd(a list object),
ss-1, ss-2, ..., ss-8, k1,..., k8, s1,..., s9, latency_s, io_mb

DF2 (qs.csv)
appid, tid, qid, qs_id, qs_lqp(json), qs_pqp(json), initial_part_size,
im-1, im-2, pd(a list object), ss-1, ss-2, ..., ss-8, k1,..., k8, s1,..., s9,
latency_s, io_mb, ana_latency_s

"""
import argparse

from udao_trace.parser.spark_parser import SparkParser
from udao_trace.utils import BenchmarkType
from udao_trace.utils.logging import _get_logger

if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser(description="Spark Trace Collection Script")
    parser.add_argument("--header", type=str,
                        default="./spark_collector/tpch100/lhs_22x2273")
    parser.add_argument("--benchmark_type", type=str, default="TPCH",
                        help="Type of benchmark (e.g., TPCH)",)
    parser.add_argument("--scale_factor", type=int, default=100,
                        help="Scale factor of the benchmark")
    parser.add_argument("--upto", type=int, default=10,
                        help="Name of the traces for each template")
    parser.add_argument("--n_processes", type=int, default=1,
                        help="number of parallel processors for parsing traces",)
    # fmt: on

    args = parser.parse_args()

    logger = _get_logger(__name__)
    sp = SparkParser(
        benchmark_type=BenchmarkType[args.benchmark_type],
        scale_factor=args.scale_factor,
        logger=logger,
    )
    templates = sp.benchmark.templates
    df_q, df_qs = sp.parse(
        header=args.header,
        templates=templates,
        upto=args.upto,
        n_processes=args.n_processes,
    )
