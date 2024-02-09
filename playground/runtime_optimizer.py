from pathlib import Path

from udao_spark.optimizer.atomic_optimizer import AtomicOptimizer
from udao_spark.optimizer.runtime_optimizer import RuntimeOptimizer
from udao_spark.optimizer.utils import get_ag_meta
from udao_spark.utils.params import QType, get_runtime_optimizer_parameters
from udao_trace.configuration import SparkConf
from udao_trace.utils.logging import logger

logger.setLevel("INFO")

if __name__ == "__main__":
    params = get_runtime_optimizer_parameters().parse_args()
    logger.info(f"get parameters: {params}")
    bm = params.benchmark
    hp_choice, graph_choice = params.hp_choice, params.graph_choice
    ag_sign = params.ag_sign
    infer_limit = params.infer_limit
    infer_limit_batch_size = params.infer_limit_batch_size

    R_Q: QType = "q_all"
    R_QS: QType = "qs_pqp_runtime"

    ag_meta_dict = {
        q_type: get_ag_meta(
            bm,
            hp_choice,
            graph_choice,
            q_type,
            ag_sign,
            infer_limit,
            infer_limit_batch_size,
        )
        for q_type in [R_Q, R_QS]
    }
    decision_variables_dict = {
        R_Q: ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9"]
        + ["s10", "s11"],  # THETA_P + THETA_S
        R_QS: ["s10", "s11"],  # THETA_S
    }

    base_dir = Path(__file__).parent
    spark_conf = SparkConf(str(base_dir / "assets/spark_configuration_aqe_on.json"))

    optimizer_dict = {
        q_type: AtomicOptimizer(
            bm=bm,
            model_sign=ag_meta_dict[q_type]["model_sign"],
            graph_model_params_path=ag_meta_dict[q_type]["model_params_path"],
            graph_weights_path=ag_meta_dict[q_type]["graph_weights_path"],
            q_type=q_type,
            data_processor_path=ag_meta_dict[q_type]["data_processor_path"],
            spark_conf=spark_conf,
            decision_variables=decision_variables_dict[q_type],
            ag_path=ag_meta_dict[q_type]["ag_path"],
        )
        for q_type in [R_Q, R_QS]
    }

    ro = RuntimeOptimizer(
        ro_q=optimizer_dict[R_Q],
        ro_qs=optimizer_dict[R_QS],
    )

    # ro.sanity_check()

    ro.setup_server(host="0.0.0.0", port=12345, debug=True)
