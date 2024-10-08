from argparse import ArgumentParser

from udao_trace.collector.SparkCollector import SparkCollector
from udao_trace.utils import BenchmarkType, ClusterName
from udao_trace_examples.params import get_collector_parser


def get_parser() -> ArgumentParser:
    # fmt: off
    parser = get_collector_parser()
    parser.add_argument("--range-start", type=int, default=-1,
        help="Start of the range (inclusive)")
    parser.add_argument("--range-end", type=int, default=-1,
        help="End of the range (inclusive)")
    # fmt: on
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()

    if args.parametric_bash_file != "assets/run_spark_collect_job.sh":
        raise ValueError(
            "parametric_bash_file must be "
            "'assets/run_spark_collect_job.sh' "
            "for job trace collection"
        )
    if not args.benchmark_type.startswith("JOB"):
        raise ValueError("benchmark_type must be 'JOB' for job trace collection")

    spark_collector = SparkCollector(
        knob_meta_file=args.knob_meta_file,
        benchmark_type=BenchmarkType[args.benchmark_type],
        scale_factor=args.scale_factor,
        cluster_name=ClusterName[args.cluster_name],
        parametric_bash_file=args.parametric_bash_file,
        header=args.trace_header,
        debug=args.debug,
    )

    if args.default:
        spark_collector.start_default(
            n_processes=args.n_processes, cluster_cores=args.cluster_cores
        )
    else:
        spark_collector.start_lhs_job(
            cluster_cores=args.cluster_cores,
            seed=args.seed,
            n_processes=args.n_processes,
            query_range=(args.range_start, args.range_end),
        )
