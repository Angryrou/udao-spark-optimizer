import argparse

from udao_trace.collector.SparkCollector import SparkCollector
from udao_trace.utils import BenchmarkType, ClusterName

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spark Trace Collection Script")
    parser.add_argument(
        "--knob_meta_file",
        type=str,
        default="assets/spark_configuration_aqe_on.json",
        help="Path to the knob metadata file",
    )
    parser.add_argument(
        "--benchmark_type",
        type=str,
        default="TPCH",
        help="Type of benchmark (e.g., TPCH)",
    )
    parser.add_argument(
        "--scale_factor", type=int, default=100, help="Scale factor of the benchmark"
    )
    parser.add_argument(
        "--cluster_name", type=str, default="HEX1", help="Name of the cluster"
    )
    parser.add_argument(
        "--parametric_bash_file",
        type=str,
        default="assets/run_spark_collect.sh",
        help="Path to the parametric bash file",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "--default", action="store_true", help="Run default configurations"
    )
    parser.add_argument(
        "--n_data_per_template",
        type=int,
        default=10,
        help="Number of data points per template, 2273 for TPCH, 490 for TPCDS",
    )
    parser.add_argument(
        "--n_processes",
        type=int,
        default=16,
        help="Number of processes for parallel execution",
    )
    parser.add_argument(
        "--cluster_cores", type=int, default=150, help="Total available cluster cores"
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for randomization")

    args = parser.parse_args()

    spark_collector = SparkCollector(
        knob_meta_file=args.knob_meta_file,
        benchmark_type=BenchmarkType[args.benchmark_type],
        scale_factor=args.scale_factor,
        cluster_name=ClusterName[args.cluster_name],
        parametric_bash_file=args.parametric_bash_file,
        header="spark_collector",
        debug=args.debug,
    )

    if args.default:
        spark_collector.start_default(
            n_processes=args.n_processes, cluster_cores=args.cluster_cores
        )
    else:
        spark_collector.start_lhs(
            n_data_per_template=args.n_data_per_template,
            n_processes=args.n_processes,
            cluster_cores=args.cluster_cores,
            seed=args.seed,
        )
