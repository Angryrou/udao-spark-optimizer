from pathlib import Path

from udao_spark.data.extractors.injection_extractor import (
    get_non_decision_inputs_for_qs_compile_dict,
)
from udao_spark.optimizer.hierarchical_optimizer import HierarchicalOptimizer
# from udao_spark.optimizer.hierarchical_optimizer import HierarchicalOptimizer
from udao_spark.optimizer.utils import get_ag_meta
from udao_spark.utils.params import QType, get_compile_time_optimizer_parameters
from udao_trace.configuration import SparkConf
from udao_trace.utils import BenchmarkType
from udao_trace.utils.logging import logger
from udao_trace.workload import Benchmark

logger.setLevel("INFO")

if __name__ == "__main__":
    # Initialize InjectionExtractor
    params = get_compile_time_optimizer_parameters().parse_args()
    logger.info(f"get parameters: {params}")

    bm = params.benchmark
    q_type: QType = params.q_type
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
        params.ag_time_limit,
    )

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
        ag_path=ag_meta["ag_path"],
        clf_json_path=None
        if params.disable_failure_clf
        else str(base_dir / f"assets/{bm}_valid_clf_meta.json"),
        clf_recall_xhold=params.clf_recall_xhold,
    )

    # Prepare traces
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

    # Compile time QS logical plans from CBO estimation (a list of LQP-sub)
    is_oracle = q_type == "qs_lqp_runtime"
    use_ag = not params.use_mlp

    ## to save query info
    # stage_list = []
    # query_id_list = []
    # for trace in raw_traces:
    #     logger.info(f"Processing {trace}")
    #     non_decision_input = get_non_decision_inputs_for_qs_compile_dict(
    #         trace, is_oracle=is_oracle
    #     )
    #     query_id = trace.split(f"{bm}100_")[1].split("_")[0]  # e.g. 2-1
    #     print(f"query_id is {query_id}")
    #
    #     n_stages = len(non_decision_input)
    #     query_id_list.append(query_id)
    #     stage_list.append(n_stages)
    # save_path = f"{params.save_data_header}/query_info/"
    # save_results(save_path, results=np.array(stage_list), mode="n_stages_list")
    # save_results(save_path, results=np.array(query_id_list), mode="query_id_list")

    for template, trace in zip(benchmark.templates, raw_traces):
        logger.info(f"Processing {trace}")

        query_id = trace.split(f"{bm}100_")[1].split("_")[0]  # e.g. 2-1
        print(f"query_id is {query_id}")

        # if query_id not in ["2-1"]:
        #     continue
        non_decision_input = get_non_decision_inputs_for_qs_compile_dict(
            trace, is_oracle=is_oracle
        )

        if params.moo_algo == "evo":
            param1 = params.pop_size
            param2 = params.nfe
            param3 = -1
        elif params.moo_algo == "ws":
            param1 = params.n_samples
            param2 = params.n_ws
            param3 = -1
        elif "hmooc" or "div_and_conq_moo" in params.moo_algo:
            param1 = params.n_c_samples
            param2 = params.n_p_samples
            param3 = -1
        elif params.moo_algo == "ppf":
            param1 = params.n_process
            param2 = params.n_grids
            param3 = params.n_max_iters
        elif params.moo_algo == "analyze_model_accuracy" or params.moo_algo == "test":
            param1 = -1
            param2 = -1
            param3 = -1
        else:
            raise Exception(f"algo {params.moo_algo} is not supported!")

        ag_model = {
            "ana_latency_s": params.ag_model_qs_ana_latency
            or "WeightedEnsemble_L2_FULL",
            "io_mb": params.ag_model_qs_io or "CatBoost",
        }

        po_points = hier_optimizer.solve(
            template=template,
            non_decision_input=non_decision_input,
            seed=params.seed,
            use_ag=use_ag,
            ag_model=ag_model,
            algo=params.moo_algo,
            save_data=params.save_data,
            query_id=query_id,
            sample_mode=params.sample_mode,
            param1=param1,
            param2=param2,
            param3=param3,
            time_limit=params.time_limit,
            is_oracle=is_oracle,
            save_data_header=params.save_data_header,
            is_query_control=params.set_query_control,
        )
