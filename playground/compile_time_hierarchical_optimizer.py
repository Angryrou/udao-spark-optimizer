from pathlib import Path

from udao_spark.data.extractors.injection_extractor import InjectionExtractor
from udao_spark.optimizer.hierarchical_optimizer import HierarchicalOptimizer
from udao_trace.configuration import SparkConf
from udao_trace.utils import BenchmarkType
from udao_trace.utils.logging import logger

logger.setLevel("INFO")
if __name__ == "__main__":
    # Initialize InjectionExtractor

    benchmark = BenchmarkType.TPCH
    scale_factor = 100
    oracle = False
    ie = InjectionExtractor(
        benchmark_type=BenchmarkType.TPCH, scale_factor=scale_factor, logger=logger
    )

    # Prepare traces

    base_dir = Path(__file__).parent
    sample_header = str(base_dir / "assets/samples")
    raw_traces = [
        f"{sample_header}/tpch100_{q}-1_1,1g,16,16,48m,200,true,0.6,"
        f"64MB,0.2,0MB,10MB,200,256MB,5,128MB,4MB,0.2,1024KB"
        f"_application_1701736595646_{2556 + q}.json"
        for q in range(1, 23)
    ]
    for trace in raw_traces:
        print(trace)
        if not Path(trace).exists():
            print(f"{trace} does not exist")
            raise FileNotFoundError(f"{trace} does not exist")

    # Initialize HierarchicalOptimizer and Model
    optimizer_preparation_header = base_dir / "assets/optimizer_preparation"
    model_sign = "graph_avg"  # graph_gtn will be ready soon.
    if oracle:
        header = optimizer_preparation_header / "qs_lqp_runtime/ea0378f56dcf"
        model_params_path = str(
            header / "graph_avg_96fd41972e88/model_struct_params.json"
        )
        weights_path = str(
            header
            / "graph_avg_96fd41972e88"
            / "learning_9c49bcdcb630"
            / "99-val_latency_s_WMAPE=0.297-val_io_mb_WMAPE=0.191"
            "-val_ana_latency_s_WMAPE=0.234.ckpt"
        )
        data_processor_path = str(header / "data_processor.pkl")
    else:
        header = optimizer_preparation_header / "qs_lqp_compile/ea0378f56dcf"
        model_params_path = str(
            header / "graph_avg_01a990cd1c09/model_struct_params.json"
        )
        weights_path = str(
            header
            / "graph_avg_01a990cd1c09"
            / "learning_9c49bcdcb630"
            / "99-val_latency_s_WMAPE=0.441-val_io_mb_WMAPE=0.352"
            "-val_ana_latency_s_WMAPE=0.408.ckpt"
        )
        data_processor_path = str(header / "data_processor.pkl")

    spark_conf = SparkConf(str(base_dir / "assets/spark_configuration_aqe_on.json"))
    decision_variables = (
        ["k1", "k2", "k3", "k4", "k5", "k6", "k7", "k8"]
        + ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9"]
        + ["s10", "s11"]
    )
    hier_optimizer = HierarchicalOptimizer(
        model_sign,
        model_params_path,
        weights_path,
        data_processor_path,
        spark_conf,
        decision_variables,
    )

    # Compile time QS logical plans from CBO estimation (a list of LQP-sub)

    for trace in raw_traces:
        logger.info(f"Processing {trace}")
        non_decision_input = ie.get_qs_lqp(trace, is_oracle=oracle)
        po_points = hier_optimizer.solve(non_decision_input)
