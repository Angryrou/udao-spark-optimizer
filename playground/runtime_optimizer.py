from pathlib import Path

from udao_spark.optimizer.runtime_optimizer import R_Q, R_QS, RuntimeOptimizer
from udao_spark.optimizer.utils import get_ag_meta
from udao_spark.utils.params import get_runtime_optimizer_parameters
from udao_trace.configuration import SparkConf
from udao_trace.utils.logging import logger

logger.setLevel("INFO")

if __name__ == "__main__":
    params = get_runtime_optimizer_parameters().parse_args()
    logger.info(f"get parameters: {params}")
    bm, seed = params.benchmark, params.seed
    hp_choice, graph_choice = params.hp_choice, params.graph_choice
    ag_sign = params.ag_sign
    infer_limit = params.infer_limit
    infer_limit_batch_size = params.infer_limit_batch_size

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

    ro = RuntimeOptimizer.from_params(
        bm, ag_meta_dict, spark_conf, decision_variables_dict, seed
    )

    if params.sanity_check:
        ro.sanity_check()
    else:
        ro.setup_server(host="0.0.0.0", port=12345, debug=True)
