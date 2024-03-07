from pathlib import Path

import numpy as np
import pandas as pd

from udao_spark.utils.analyzer import get_non_decision_inputs
from udao_spark.utils.params import get_ag_parameters
from udao_trace.configuration import SparkConf

if __name__ == "__main__":
    params = get_ag_parameters().parse_args()
    if params.q_type != "q_compile":
        raise ValueError(f"Diagnosing {params.q_type} is not our focus.")
    if params.hp_choice != "tuned-0215":
        raise ValueError(f"hp_choice {params.hp_choice} is not supported.")
    if params.graph_choice != "gtn":
        raise ValueError(f"graph_choice {params.graph_choice} is not supported.")

    # theta includes 19 decision variables
    decision_variables = (
        ["k1", "k2", "k3", "k4", "k5", "k6", "k7", "k8"]
        + ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9"]
        + ["s10", "s11"]
    )
    base_dir = Path(__file__).parent
    df, ag_server = get_non_decision_inputs(
        base_dir=base_dir,
        params=params,
        decision_vars=decision_variables,
    )
    # example of setting different theta for TPCH q1-1
    target_query = "q1-1"

    # 1. get non decision inputs from input_dict
    target_df = df.loc[[target_query]]
    if target_df.empty:
        raise ValueError(f"target query {target_query} not found")

    # 2. prepare theta for the target query
    base_dir = Path(__file__).parent
    spark_conf = SparkConf(str(base_dir / "assets/spark_configuration_aqe_on.json"))
    # note that the knob values are SCALED to integer from the raw traces
    # more details are in assets/spark_configuration_aqe_on.json
    theta_anchor = [1, 1, 16, 4, 5, 1, 1, 75, 5, 6, 32, 32, 50, 4, 80, 4, 4, 35, 6]
    theta_lower_bounds = np.array(spark_conf.knob_min)
    theta_upper_bounds = np.array(spark_conf.knob_max)
    if np.any(np.array(theta_anchor) < theta_lower_bounds) or np.any(
        np.array(theta_anchor) > theta_upper_bounds
    ):
        raise Exception("invalid values in theta_anchor")

    target_df[decision_variables] = theta_anchor
    # you can play with the following three knobs
    # k1 (#cores/exec),
    # k2 (mem-G/cores) => mem-G/exec = k1 * k2
    # k3 (#exec) and fix other knobs.
    # total cores = k1 * k3
    # total memory = k1 * k2 * k3
    # e.g.,
    k1k2k3_list = [
        [1, 1, 4],  # 1*4=4 cores, 1*1*4=4GB, 4 exec (minimum resources setting)
        [1, 1, 16],  # 1*16=16 core, 1*1*16=16GB, 16 exec (default resources setting)
        [5, 4, 16],  # 5*16=80 cores, 5*4*16=320GB, 16 exec (maximum resources setting)
    ]
    target_df = pd.DataFrame(
        np.tile(target_df.values, (len(k1k2k3_list), 1)), columns=target_df.columns
    )
    target_df[["k1", "k2", "k3"]] = k1k2k3_list
    if np.any(target_df[decision_variables] < theta_lower_bounds) or np.any(
        target_df[decision_variables] > theta_upper_bounds
    ):
        raise Exception("invalid decision variables in target_df")

    # 3. get the latency predictor
    lat_predictor = ag_server.predictors["latency_s"]
    lat_predictor_path = lat_predictor.path
    print(f"model lat_predictor_path: {lat_predictor_path}")

    # 4. predict the latency
    lat_pred = lat_predictor.predict(target_df, model="WeightedEnsemble_L2")

    print(lat_pred)
