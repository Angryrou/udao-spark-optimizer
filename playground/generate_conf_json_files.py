# Copyright (c) 2020 École Polytechnique
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Author(s): Qi FAN <qi dot fan at polytechnique dot edu>
#
# Description: TODO
#
# Created at 16/02/2024

import os
import json
import numpy as np

from udao_spark.optimizer.utils import weighted_utopia_nearest_impl, utopia_nearest
from typing import Tuple

len_theta_c = 8
len_theta_p = 9
len_theta_s = 2
len_theta_per_qs = len_theta_c + len_theta_p + len_theta_s

def get_recommendations(po_objs, po_confs, mode="wun", weights=np.array([0.1, 0.9])):
    if mode == "wun":
        objs, conf = weighted_utopia_nearest(po_objs, po_confs, weights)
    elif mode == "un":
        objs, conf = utopia_nearest(po_objs, po_confs)
    else:
        raise Exception(f"Recommendation mode {mode} is not supported!")

    return objs, conf

def weighted_utopia_nearest(
    pareto_objs: np.ndarray, pareto_confs: np.ndarray, weights: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    return the Pareto point that is closest to the utopia point
    in a weighted distance function
    """
    n_pareto = pareto_objs.shape[0]
    assert n_pareto > 0
    if n_pareto == 1:
        # (2,), (n, 2)
        return pareto_objs[0], pareto_confs[0]

    utopia = np.zeros_like(pareto_objs[0])
    min_objs, max_objs = pareto_objs.min(0), pareto_objs.max(0)
    pareto_norm = (pareto_objs - min_objs) / (max_objs - min_objs)
    # fixme: internal weights
    # weights = np.array([1, 1])
    # weights = np.array([0.7, 0.3])
    pareto_weighted_norm = pareto_norm * weights
    # check the speed comparison: https://stackoverflow.com/a/37795190/5338690
    dists = np.sum((pareto_weighted_norm - utopia) ** 2, axis=1)
    wun_id = np.argmin(dists)

    picked_pareto = pareto_objs[wun_id]
    picked_confs = pareto_confs[wun_id]

    return picked_pareto, picked_confs

def generate_json_file(methods, queries, stages, cpu_mode, sample_mode, save_data_header, save_json_header, evo_setting, ws_setting, ppf_setting,
                                   div_moo_setting, is_query_control=False, rec_mode="wun", is_oracle=False, model_name="mlp", weights=None, is_test_end=True):
    knob_name_dict = read_knob_name()

    # if is_oracle or is_test_end:
    #     methods = ["div_and_conq_moo%B"]
    # else:
    #     if is_query_control:
    #         methods = ["ws", "evo", "ppf"]
    #     else:
    #         methods = ["ws", "evo", "ppf", "div_and_conq_moo%B"]

    for query_id, n_stages in zip(queries, stages):
        for method in methods:
            if method == "ws":
                data_path = f"{save_data_header}/query_control_{is_query_control}/latest_model_{cpu_mode}/{model_name}/oracle_{is_oracle}/{method}/" \
                            f"{ws_setting}/time_-1/query_{query_id}_n_{n_stages}"
            elif method == "evo":
                data_path = f"{save_data_header}/query_control_{is_query_control}/latest_model_{cpu_mode}/{model_name}/oracle_{is_oracle}/{method}/" \
                            f"{evo_setting}/time_-1/query_{query_id}_n_{n_stages}"
            elif method == "ppf":
                data_path = f"{save_data_header}/query_control_{is_query_control}/latest_model_{cpu_mode}/{model_name}/oracle_{is_oracle}/{method}/" \
                            f"{ppf_setting}/time_-1/query_{query_id}_n_{n_stages}"
            else:
                dag_opt_algo = method.split("%")[1]
                data_path = f"{save_data_header}/query_control_False/latest_model_{cpu_mode}/{model_name}/oracle_{is_oracle}/{method}/" \
                            f"{div_moo_setting}/time_-1/query_{query_id}_n_{n_stages}/{sample_mode}/{dag_opt_algo}"

            theta = np.loadtxt(f"{data_path}/Theta_all.txt", delimiter=" ", dtype=str).reshape(-1, len_theta_per_qs)
            F = np.loadtxt(f"{data_path}/F.txt").reshape(-1, 2)

            if is_query_control:
                all_confs = theta
            else:
                theta_split = np.split(theta, n_stages)
                all_confs = np.hstack(theta_split)
                print()

            assert all_confs.shape[0] == F.shape[0]

            rec_F, rec_conf = get_recommendations(F, all_confs, mode=rec_mode, weights=weights)

            conf_dict = transform_conf_arr_to_json(rec_conf, knob_name_dict, is_query_control, n_stages)
            save_json_path = f"{save_json_header}/query_control_{is_query_control}/{rec_mode}/{method}"

            save_json(save_json_path, conf_dict, query_id)

            print()


def read_knob_name():
    conf_file = "./assets/spark_configuration_aqe_on.json"
    with open(conf_file) as f:
        d = json.load(f)
        print(d)

    knob_name_dict = {x["id"]: x["name"] for x in d}

    return knob_name_dict


def transform_conf_arr_to_json(rec_conf, knob_name_dict, is_query_control, n_stages):
    if is_query_control:
        assert rec_conf.shape[0] == len_theta_c + len_theta_p + len_theta_s
        theta_c = rec_conf[:len_theta_c]
        theta_c_knob_names = [knob_name_dict[f"k{i}"] for i in range(1, len_theta_c + 1)]
        theta_c_dict = {v: conf for v, conf in zip(theta_c_knob_names, theta_c)}

        theta_p = rec_conf[len_theta_c: len_theta_c + len_theta_p]
        theta_p_knob_names = [knob_name_dict[f"s{i}"] for i in range(1, len_theta_p + 1)]
        theta_p_dict = {v: conf for v, conf in zip(theta_p_knob_names, theta_p)}

        theta_s = rec_conf[len_theta_c + len_theta_p: len_theta_c + len_theta_p + len_theta_s]
        theta_s_knob_names = [knob_name_dict[f"s{i}"] for i in range(len_theta_p + 1, len_theta_p + len_theta_s + 1)]
        theta_s_dict = {v: conf for v, conf in zip(theta_s_knob_names, theta_s)}
        conf_dict = {"theta_c": theta_c_dict,
                     "theta_p": theta_p_dict,
                     "theta_s": theta_s_dict
                     }
    else:
        assert rec_conf.shape[0] == (len_theta_c + len_theta_p + len_theta_s) * n_stages
        split_rec_conf = np.split(rec_conf, n_stages)
        qs_keys = [f"qs{i}" for i in range(n_stages)]

        theta_c = np.unique(np.array(split_rec_conf)[:, :len_theta_c], axis=0).squeeze()
        assert theta_c.shape[0] == len_theta_c
        theta_c_knob_names = [knob_name_dict[f"k{i}"] for i in range(1, len_theta_c + 1)]
        theta_c_dict = {v: conf for v, conf in zip(theta_c_knob_names, theta_c)}

        theta_p_all_qs = np.array(split_rec_conf)[:, len_theta_c: len_theta_c + len_theta_p]
        theta_s_all_qs = np.array(split_rec_conf)[:, len_theta_c + len_theta_p: len_theta_per_qs]

        runtime_theta_dict = {k:{} for k in qs_keys}
        for i, k in enumerate(qs_keys):
            theta_p = theta_p_all_qs[i, :]
            theta_p_knob_names = [knob_name_dict[f"s{i}"] for i in range(1, len_theta_p + 1)]
            theta_p_dict = {v: conf for v, conf in zip(theta_p_knob_names, theta_p)}

            theta_s = theta_s_all_qs[i, :]
            theta_s_knob_names = [knob_name_dict[f"s{i}"] for i in
                                  range(len_theta_p + 1, len_theta_p + len_theta_s + 1)]
            theta_s_dict = {v: conf for v, conf in zip(theta_s_knob_names, theta_s)}
            runtime_theta_dict[k] = {"theta_p": theta_p_dict, "theta_s": theta_s_dict}

        conf_dict = {"theta_c": theta_c_dict, "runtime_theta": runtime_theta_dict}

    return conf_dict


def save_json(save_json_path, data, query_id):
    if not os.path.exists(save_json_path):
        os.makedirs(save_json_path, exist_ok=True)

    with open(f'{save_json_path}/query_{query_id}.json', 'w') as fp:
        json.dump(data, fp, indent=4, separators=(',', ': '))

if __name__ == "__main__":
    sample_mode1 = "random"
    sample_mode2 = "grid"
    cpu_mode1 = "cpu" # "cpu" or "cuda"
    cpu_mode2 = "cuda"  # "cpu" or "cuda"
    temp1 = "tpch"
    temp2 = "tpcds"
    model_name1 = "mlp"
    model_name2 = "ag"
    is_oracle = False
    is_query_control = False
    rec_mode = "wun"
    weights = np.array([0.7, 0.3])

    queries_tpch = [f"{i}-1" for i in np.arange(1, 23)]
    stages_tpch = [3, 14, 5, 5, 11, 2, 11, 13, 12, 7, 10, 5, 5, 4, 7, 7, 7, 9, 5, 14, 12, 7]

    # save_data_header = f"./output/0218test/oracle_{is_oracle}"
    save_data_header = f"./output/0218test/{temp2}_traces"
    save_json_header = f"./output/0218test/{temp2}_traces/{model_name2}/oracle_{is_oracle}/{weights[0]}_{weights[1]}/configurations_json"
    # save_data_header = f"./output/0221expr/{temp2}"
    # save_json_header = f"./output/0221expr/{temp2}/{model_name1}/configurations_json"

    # for TPCDS
    query_id_path = f"{save_data_header}/query_info/query_id_list.txt"
    n_stages_path = f"{save_data_header}/query_info/n_stages_list.txt"

    queries_tpcds = np.loadtxt(f"{query_id_path}", delimiter=" ", dtype=str).tolist()
    stages_tpcds = np.loadtxt(f"{n_stages_path}").astype(int).tolist()

    evo_setting = "100_500"
    ws_setting = "100000_11"
    # ppf_setting = "1_4_4" # TPCH
    ppf_setting = "1_2_2"  # TPCDS
    # if temp1:
    #     div_moo_setting = "64_128"
    # else:
    #     div_moo_setting = "64_256"
    div_moo_setting1 = "128_256"  # tpch
    div_moo_setting2 = "128_32"  # tpcds

    # read_knob_name()

    # for query 5-1
    # query_5 = [queries[4]]
    # stages_5 = [stages[4]]

    # generate_json_file(queries, stages,
    #                    cpu_mode=cpu_mode1,
    #                    sample_mode=sample_mode2,
    #                    save_data_header=save_data_header,
    #                    save_json_header=save_json_header,
    #                    evo_setting=evo_setting,
    #                    ws_setting=ws_setting,
    #                    ppf_setting=ppf_setting,
    #                    div_moo_setting=div_moo_setting2,
    #                    is_query_control=is_query_control,
    #                    rec_mode=rec_mode,
    #                    model_name=model_name2,
    #                    weights=weights,
    #                    is_oracle=is_oracle,
    #                    )


    # generate configuration files with more choices on resources
    # TPCH
    # save_data_header_tpch = "./output/0218test/tpch_traces/new_test_end"
    # save_data_header_tpcds = "./output/0218test/tpcds_traces/new_test_end"

    save_data_header_tpch = "./output/0218test/oracle_True/tpch_traces/new_test_end"
    save_data_header_tpcds = "./output/0218test/oracle_True/tpcds_traces/new_test_end"
    div_moo_setting_tpch = "256_256"  # tpch
    div_moo_setting_tpcds1 = "256_32"  # tpcds
    div_moo_setting_tpcds2 = "512_32"  # tpcds

    save_json_header_tpch = f"./output/0218test/tpch_traces/new_test_end/{model_name2}/oracle_{is_oracle}/{weights[0]}_{weights[1]}/configurations_json"
    save_json_header_tpcds1 = f"./output/0218test/tpcds_traces/new_test_end/{model_name2}/oracle_{is_oracle}/{weights[0]}_{weights[1]}/" \
                             f"{div_moo_setting_tpcds1}/configurations_json"
    save_json_header_tpcds2 = f"./output/0218test/tpcds_traces/new_test_end/{model_name2}/oracle_{is_oracle}/{weights[0]}_{weights[1]}/" \
                             f"{div_moo_setting_tpcds2}/configurations_json"
    # generate_json_file(queries=queries_tpcds,
    #                    stages=stages_tpcds,
    #                    cpu_mode=cpu_mode1,
    #                    sample_mode=sample_mode2,
    #                    save_data_header=save_data_header_tpcds,
    #                    save_json_header=save_json_header_tpcds1,
    #                    evo_setting=evo_setting,
    #                    ws_setting=ws_setting,
    #                    ppf_setting=ppf_setting,
    #                    div_moo_setting=div_moo_setting_tpcds1,
    #                    is_query_control=False,
    #                    rec_mode=rec_mode,
    #                    model_name=model_name2,
    #                    weights=weights,
    #                    is_oracle=is_oracle,
    #                    is_test_end=True,
    #                    )

    # generate_json_file(queries=queries_tpcds,
    #                    stages=stages_tpcds,
    #                    cpu_mode=cpu_mode1,
    #                    sample_mode=sample_mode2,
    #                    save_data_header=save_data_header_tpcds,
    #                    save_json_header=save_json_header_tpcds2,
    #                    evo_setting=evo_setting,
    #                    ws_setting=ws_setting,
    #                    ppf_setting=ppf_setting,
    #                    div_moo_setting=div_moo_setting_tpcds2,
    #                    is_query_control=False,
    #                    rec_mode=rec_mode,
    #                    model_name=model_name2,
    #                    weights=weights,
    #                    is_oracle=is_oracle,
    #                    is_test_end=True,
    #                    )
    #
    # generate_json_file(queries=queries_tpch,
    #                    stages=stages_tpch,
    #                    cpu_mode=cpu_mode1,
    #                    sample_mode=sample_mode2,
    #                    save_data_header=save_data_header_tpch,
    #                    save_json_header=save_json_header_tpch,
    #                    evo_setting=evo_setting,
    #                    ws_setting=ws_setting,
    #                    ppf_setting=ppf_setting,
    #                    div_moo_setting=div_moo_setting_tpch,
    #                    is_query_control=False,
    #                    rec_mode=rec_mode,
    #                    model_name=model_name2,
    #                    weights=weights,
    #                    is_oracle=is_oracle,
    #                    is_test_end=True,
    #                    )

    # save_data_header_rerun_tpch = "./output/0218test/tpch_traces/rerun_fail_queries"
    # save_json_header_rerun_tpch = f"./output/0218test/tpch_traces/rerun_fail_queries/{model_name2}/oracle_{is_oracle}/{weights[0]}_{weights[1]}/configurations_json"
    # methods = ["evo"]
    # queries_rerun_tpch = [queries_tpch[8]]
    # stages_rerun_tpch = [stages_tpch[8]]
    # generate_json_file(queries=queries_rerun_tpch,
    #                    stages=stages_rerun_tpch,
    #                    methods=methods,
    #                    cpu_mode=cpu_mode1,
    #                    sample_mode=sample_mode2,
    #                    save_data_header=save_data_header_rerun_tpch,
    #                    save_json_header=save_json_header_rerun_tpch,
    #                    evo_setting=evo_setting,
    #                    ws_setting=ws_setting,
    #                    ppf_setting=ppf_setting,
    #                    div_moo_setting=div_moo_setting1,
    #                    is_query_control=is_query_control,
    #                    rec_mode=rec_mode,
    #                    model_name=model_name2,
    #                    weights=weights,
    #                    is_oracle=is_oracle,
    #                    is_test_end=True,
    #                    )

    save_data_header_more_preference_tpch = "./output/0218test"
    save_json_header_more_preference_tpch = f"./output/0218test/tpch_traces/more_preferences/{model_name2}/oracle_{is_oracle}/{weights[0]}_{weights[1]}/configurations_json"

    save_data_header_more_preference_tpcds = "./output/0218test/tpcds_traces"
    save_json_header_more_preference_tpcds = f"./output/0218test/tpcds_traces/more_preferences/{model_name2}/oracle_{is_oracle}/{weights[0]}_{weights[1]}/configurations_json"

    methods = ["div_and_conq_moo%B"]
    # queries_rerun_tpch = [queries_tpch[8]]
    # stages_rerun_tpch = [stages_tpch[8]]
    generate_json_file(queries=queries_tpch,
                       stages=stages_tpch,
                       methods=methods,
                       cpu_mode=cpu_mode1,
                       sample_mode=sample_mode2,
                       save_data_header=save_data_header_more_preference_tpch,
                       save_json_header=save_json_header_more_preference_tpch,
                       evo_setting=evo_setting,
                       ws_setting=ws_setting,
                       ppf_setting=ppf_setting,
                       div_moo_setting=div_moo_setting1,
                       is_query_control=is_query_control,
                       rec_mode=rec_mode,
                       model_name=model_name2,
                       weights=weights,
                       is_oracle=is_oracle,
                       is_test_end=True,
                       )

    generate_json_file(queries=queries_tpcds,
                       stages=stages_tpcds,
                       methods=methods,
                       cpu_mode=cpu_mode1,
                       sample_mode=sample_mode2,
                       save_data_header=save_data_header_more_preference_tpcds,
                       save_json_header=save_json_header_more_preference_tpcds,
                       evo_setting=evo_setting,
                       ws_setting=ws_setting,
                       ppf_setting=ppf_setting,
                       div_moo_setting=div_moo_setting2,
                       is_query_control=is_query_control,
                       rec_mode=rec_mode,
                       model_name=model_name2,
                       weights=weights,
                       is_oracle=is_oracle,
                       is_test_end=True,
                       )

