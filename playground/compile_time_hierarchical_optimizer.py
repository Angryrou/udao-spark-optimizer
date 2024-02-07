from pathlib import Path

from udao_spark.data.extractors.injection_extractor import InjectionExtractor
from udao_spark.optimizer.hierarchical_optimizer import HierarchicalOptimizer
from udao_spark.optimizer.utils import get_ag_meta
from udao_spark.utils.params import get_compile_time_optimizer_parameters
from udao_trace.configuration import SparkConf
from udao_trace.utils import BenchmarkType
from udao_trace.utils.logging import logger

logger.setLevel("INFO")


if __name__ == "__main__":
    # Initialize InjectionExtractor
    params = get_compile_time_optimizer_parameters().parse_args()
    logger.info(f"get parameters: {params}")

    bm, q_type = params.benchmark, params.q_type
    hp_choice, graph_choice = params.hp_choice, params.graph_choice
    ag_sign = params.ag_sign
    infer_limit = params.infer_limit
    infer_limit_batch_size = params.infer_limit_batch_size

    if q_type not in ["qs_lqp_compile", "qs_lqp_runtime"]:
        raise ValueError(
            f"q_type {q_type} is not supported for "
            f"compile time hierarchical optimizer"
        )

    ag_meta = get_ag_meta(
        bm,
        hp_choice,
        graph_choice,
        q_type,
        ag_sign,
        infer_limit,
        infer_limit_batch_size,
    )
    ag_path = ag_meta["ag_path"]
    logger.info(f"ag_path: {ag_path}")

    # Initialize HierarchicalOptimizer and Model
    base_dir = Path(__file__).parent
    spark_conf = SparkConf(str(base_dir / "assets/spark_configuration_aqe_on.json"))
    decision_variables = (
        ["k1", "k2", "k3", "k4", "k5", "k6", "k7", "k8"]
        + ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9"]
        + ["s10", "s11"]
    )

    hier_optimizer = HierarchicalOptimizer(
        bm=bm,
        model_sign=ag_meta["model_sign"],
        graph_model_params_path=ag_meta["model_params_path"],
        graph_weights_path=ag_meta["graph_weights_path"],
        q_type=params.q_type,
        data_processor_path=ag_meta["data_processor_path"],
        spark_conf=spark_conf,
        decision_variables=decision_variables,
        ag_path=ag_path,
    )

    # Prepare traces
    ie = InjectionExtractor(
        benchmark_type=BenchmarkType[params.benchmark.upper()],
        scale_factor=params.scale_factor,
        logger=logger,
    )
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

    # Compile time QS logical plans from CBO estimation (a list of LQP-sub)
    is_oracle = q_type == "qs_lqp_runtime"
    use_ag = not params.use_mlp
    for trace in raw_traces:
        logger.info(f"Processing {trace}")
        non_decision_input = ie.get_qs_lqp(trace, is_oracle=is_oracle)
        po_points = hier_optimizer.solve(
            non_decision_input,
            seed=params.seed,
            use_ag=use_ag,
            ag_model=params.ag_model,
        )
