from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from udao_spark.data.utils import get_ag_data
from udao_spark.model.model_server import AGServer
from udao_spark.optimizer.utils import get_ag_meta
from udao_spark.utils.params import get_ag_parameters
from udao_trace.configuration import SparkConf
from udao_trace.utils.logging import logger


def data_preparation(
    params: Namespace,
) -> Tuple[Dict[str, pd.DataFrame], AGServer]:
    if params.q_type != "q_compile":
        raise ValueError(f"Diagnosing {params.q_type} is not our focus.")
    if params.hp_choice != "tuned-0215":
        raise ValueError(f"hp_choice {params.hp_choice} is not supported.")
    if params.graph_choice != "gtn":
        raise ValueError(f"graph_choice {params.graph_choice} is not supported.")
    bm, q_type, debug = params.benchmark, params.q_type, params.debug
    ag_meta = get_ag_meta(
        bm,
        params.hp_choice,
        params.graph_choice,
        q_type,
        params.ag_sign,
        params.infer_limit,
        params.infer_limit_batch_size,
        params.ag_time_limit,
    )
    weights_path = ag_meta["graph_weights_path"]
    ret = get_ag_data(bm, q_type, debug, params.graph_choice, weights_path, if_df=True)
    data_tr, data_val, data_te = ret[
        "data"
    ]  # choose the validation data set ~5K queries
    data_query_tr, data_query_val, data_query_te = ret["data_queries"]
    data_tr[["template", "q"]] = data_query_tr.values.astype(str)
    data_tr["query_id"] = data_tr["template"] + "-" + data_tr["q"]
    data_val[["template", "q"]] = data_query_val.values.astype(str)
    data_val["query_id"] = data_val["template"] + "-" + data_val["q"]
    data_te[["template", "q"]] = data_query_te.values.astype(str)
    data_te["query_id"] = data_te["template"] + "-" + data_te["q"]

    base_dir = Path(__file__).parent
    ag_server = AGServer.from_ckp_path(
        model_sign=ag_meta["model_sign"],
        graph_model_params_path=ag_meta["model_params_path"],
        graph_weights_path=ag_meta["graph_weights_path"],
        q_type=q_type,
        ag_path=ag_meta["ag_path"],
        clf_json_path=str(base_dir / f"assets/{bm}_valid_clf_meta.json"),
        clf_recall_xhold=params.clf_recall_xhold,
    )

    return {
        "data_tr": data_tr,
        "data_val": data_val,
        "data_te": data_te,
    }, ag_server


def get_lhs_confs(spark_conf: SparkConf, n_samples: int, seed: int) -> pd.DataFrame:
    lhs_conf_raw = spark_conf.get_lhs_configurations(n_samples, seed=seed)
    lhs_conf = pd.DataFrame(
        data=spark_conf.deconstruct_configuration(lhs_conf_raw.values),
        columns=spark_conf.knob_ids,
    )
    return lhs_conf


def get_resource_knob_permutation() -> pd.DataFrame:
    # get 5 * 4 * 13 = 260 configurations combinations for k1, k2, k3
    k1 = np.arange(1, 6).astype(np.float32)
    k2 = np.arange(1, 5).astype(np.float32)
    k3 = np.arange(4, 17).astype(np.float32)
    conf_res = np.array(np.meshgrid(k1, k2, k3)).T.reshape(-1, 3)
    return pd.DataFrame(data=conf_res, columns=["k1", "k2", "k3"])


def cross_product_two_numpy(a1: np.ndarray, a2: np.ndarray) -> np.ndarray:
    idx2, idx1 = np.meshgrid(range(a2.shape[0]), range(a1.shape[0]), indexing="ij")
    idx1 = idx1.flatten()
    idx2 = idx2.flatten()
    return np.concatenate([a1[idx1], a2[idx2]], axis=1)


def get_lhs_confs_with_res_perm(
    spark_conf: SparkConf, n_samples: int, seed: int
) -> pd.DataFrame:
    # get configurations via LHS
    lhs_sampled_confs = get_lhs_confs(spark_conf, n_samples, seed=seed)
    if len(lhs_sampled_confs) < n_samples:
        logger.warning(
            f"lost {n_samples - len(lhs_sampled_confs)} "
            f"configurations via LHS due to duplication"
        )

    # get 260 configurations combinations for k1, k2, k3
    conf_res_perm = get_resource_knob_permutation()
    # replace the first 3 columns of lhs_sampled_confs with conf_res_perm
    idx_lhs, idx_res = np.meshgrid(
        range(len(lhs_sampled_confs)), range(len(conf_res_perm)), indexing="ij"
    )
    idx_lhs = idx_lhs.flatten()
    idx_res = idx_res.flatten()
    merged = np.concatenate(
        [conf_res_perm.values[idx_res], lhs_sampled_confs.values[idx_lhs, 3:]], axis=1
    )
    # get 260 * n_samples configurations
    return pd.DataFrame(data=merged, columns=lhs_sampled_confs.columns)


def get_params() -> ArgumentParser:
    parser = get_ag_parameters()
    parser.add_argument("--demo", action="store_true", help="demo mode")
    parser.add_argument(
        "--ag_model_q_latency",
        type=str,
        default=None,
        help="specific model name for AG for latency",
    )
    parser.add_argument(
        "--ag_model_q_io",
        type=str,
        default=None,
        help="specific model name for AG for IO",
    )
    return parser


if __name__ == "__main__":
    params = get_params().parse_args()
    data_all, ag_server = data_preparation(params)
    base_dir = Path(__file__).parent
    spark_conf = SparkConf(str(base_dir / "assets/spark_configuration_aqe_on.json"))
    decision_variables = (
        ["k1", "k2", "k3", "k4", "k5", "k6", "k7", "k8"]
        + ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9"]
    ) + ["s10", "s11"]
    if decision_variables != spark_conf.knob_ids:
        raise ValueError(f"invalide decision variables {spark_conf.knob_ids}")
    # choose validation dataset
    data = data_all["data_val"]
    query_ids = data["query_id"].tolist()
    print(f"total number of queries in validation set: {len(query_ids)}")
    print(data.head())

    if params.demo:
        # example of one query
        query_id = query_ids[0]

        # 1. prepare the data features for the query
        df = data.loc[data["query_id"] == query_id]
        # target n_conf_samples=10000 for each query in real runs
        n_conf_samples = 10
        # try different seeds for templates or queries to hit more coverage in real runs
        seed = 0
        lhs_confs_with_res_perm = get_lhs_confs_with_res_perm(
            spark_conf, n_conf_samples, seed=seed
        )
        # prepare data for prediction towards one query with ~(260 * 10) configurations
        target_df = pd.DataFrame(
            np.tile(df.values, (len(lhs_confs_with_res_perm), 1)), columns=df.columns
        )
        target_df[decision_variables] = lhs_confs_with_res_perm.values

        # 2. provide the predictions.
        lat_predictor = ag_server.predictors["latency_s"]
        lat_predictor_path = lat_predictor.path
        print(f"model lat_predictor_path: {lat_predictor_path}")
        # 4. predict the latency
        lat_pred = lat_predictor.predict(target_df, model=params.ag_model_q_latency)
    else:
        print("demo mode is off")
        # TODO
