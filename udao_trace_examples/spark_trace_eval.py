from argparse import ArgumentParser
from typing import Dict, List, Tuple

from udao_trace.collector.SparkCollector import SparkCollector
from udao_trace.utils import BenchmarkType, ClusterName, JsonHandler
from udao_trace_examples.params import get_collector_parser


def parse_configuration(j: Dict, fine_grained: bool) -> Dict:
    if "submit_theta" not in j:
        if fine_grained:
            config = {
                **j["theta_c"],
                **j["runtime_theta"]["qs0"]["theta_p"],
                **j["runtime_theta"]["qs0"]["theta_s"],
            }
        else:
            config = {**j["theta_c"], **j["theta_p"], **j["theta_s"]}
    else:
        config = j["submit_theta"]
    return config


def validate_and_parse_configurations(
    spark_collector: SparkCollector,
    config_header: str,
    n_data_per_template: int,
    fine_grained: bool = False,
) -> Tuple[str, List[Dict]]:
    header = spark_collector.header + "/" + config_header
    templates = spark_collector.benchmark.templates
    configurations = []
    for qid in range(1, n_data_per_template + 1):
        for template in templates:
            try:
                j = JsonHandler.load_json(f"{header}/query_{template}-{qid}.json")
                configuration = parse_configuration(j, fine_grained)
            except FileNotFoundError:
                raise ValueError(f"Configuration not found for {template}, {qid}")
            except Exception as e:
                raise e

            for c_name in spark_collector.spark_conf.knob_names:
                if c_name not in configuration:
                    raise ValueError(
                        f"Configuration for {template}, {qid} is missing {c_name}"
                    )
            configurations.append({(template, qid): configuration})
    return header, configurations


def get_eval_parser() -> ArgumentParser:
    parser = get_collector_parser()
    # fmt: off
    parser.add_argument("--configuration_header", type=str,
                        help="Header for the configurations")
    parser.add_argument("--fine_grained", action="store_true",
                        help="Use fine-grained configurations")
    parser.add_argument("--n_reps", type=int, default=3,
                        help="Number of repetitions for each configuration")
    # fmt: on

    return parser


if __name__ == "__main__":
    args = get_eval_parser().parse_args()
    if args.trace_header != "evaluations":
        raise ValueError("trace_header must be 'evaluations'")

    spark_collector = SparkCollector(
        knob_meta_file=args.knob_meta_file,
        benchmark_type=BenchmarkType[args.benchmark_type],
        scale_factor=args.scale_factor,
        cluster_name=ClusterName[args.cluster_name],
        parametric_bash_file=args.parametric_bash_file,
        header=args.trace_header,
        debug=args.debug,
    )

    header, configurations = validate_and_parse_configurations(
        spark_collector=spark_collector,
        config_header=args.configuration_header,
        n_data_per_template=args.n_data_per_template,
        fine_grained=args.fine_grained,
    )

    for i in range(args.n_reps):
        print("------ Starting evaluation", i + 1)
        spark_collector.start_eval(
            eval_header=header,
            configurations=configurations,
            n_processes=args.n_processes,
            cluster_cores=args.cluster_cores,
        )
