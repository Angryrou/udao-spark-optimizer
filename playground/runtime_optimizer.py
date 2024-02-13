from pathlib import Path

from udao_spark.optimizer.runtime_optimizer import R_Q, R_QS, RuntimeOptimizer
from udao_spark.optimizer.utils import get_ag_meta
from udao_spark.utils.logging import logger
from udao_spark.utils.params import get_runtime_optimizer_parameters
from udao_trace.configuration import SparkConf

if __name__ == "__main__":
    params = get_runtime_optimizer_parameters().parse_args()
    debug = params.debug
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
        bm,
        ag_meta_dict,
        spark_conf,
        decision_variables_dict,
        seed,
    )
    use_ag = not params.use_mlp
    if params.sanity_check:
        logger.setLevel("DEBUG")
        for file_path in [
            "assets/runtime_samples/sample_runtime_lqp.txt",
            "assets/runtime_samples/sample_runtime_qs.txt",
            "assets/runtime_samples/sample_runtime_qs2.txt",
        ]:
            if not Path(file_path).exists():
                raise FileNotFoundError(f"{file_path} does not exist")
            ro.sanity_check(
                file_path=file_path,
                use_ag=use_ag,
                ag_model=params.ag_model,
                sample_mode=params.sample_mode,
                n_samples=params.n_samples,
                moo_mode=params.moo_mode,
            )
    else:
        ro.setup_server(
            host="0.0.0.0",
            port=12345,
            debug=debug,
            use_ag=use_ag,
            ag_model=params.ag_model,
            sample_mode=params.sample_mode,
            n_samples=params.n_samples,
            moo_mode=params.moo_mode,
        )
