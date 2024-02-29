# Copyright (c) 2024 École Polytechnique
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Author(s): Qi FAN <qi dot fan at polytechnique dot edu>
#
# Description: TODO
#
# Created at 14/02/2024

from typing import Tuple
import itertools
import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os
import torch as th
from botorch.utils.multi_objective.hypervolume import Hypervolume #type: ignore
from udao_spark.optimizer.utils import save_results, is_pareto_efficient, even_weights

import seaborn as sns
import pandas as pd

def dominated_space(pareto_set: th.Tensor, ref_point: th.Tensor) -> float:
    hv = Hypervolume(ref_point=ref_point)
    volume = hv.compute(pareto_set)

    return volume


def cal_hv(F_list, nadir, utopia):
    ## calculate the Nadir point among all methods
    dominated_space_list = []
    for F in F_list:
        dominated_space = perc_dominated_space(np.array(F), nadir, utopia)
        dominated_space_list.append(dominated_space)

    return dominated_space_list

def perc_dominated_space(pareto_set: th.Tensor, ref_point: np.ndarray, utopia_point: np.ndarray) -> float:
    # for 2D minimization
    init_volume = abs(ref_point[1] - utopia_point[1]) * abs(
        ref_point[0] - utopia_point[0]
    )
    dominated_volume = dominated_space(
        -1 * th.from_numpy(pareto_set), -1 * th.from_numpy(ref_point)
    )

    return 100 * (dominated_volume / init_volume)

def get_utopia_nadir_all_algos(methods, query_id, n_stages, sample_mode, cpu_mode="cpu",
                               n_c_samples_list=[-1], n_p_samples_list=[-1],
                               evo_setting="100_1000",
                               ws_setting="1000000_11",
                               ppf_setting="1_1_1",
                               save_data_header="",
                               is_query_control=False,
                               is_oracle=False,
                               model_name="mlp"):

    # new HV analaysis: to have consistent reference points for all algorithms and all settings
    F_list = []
    for method in methods:
        if method == "evo":
            # data_path = f"./output/output/test/latest_model_{cpu_mode}/ag/{method}/100_1000/time_-1/query_{query_id}_n_{n_stages}"
            # data_path = f"./output/test/query_control_True/latest_model_{cpu_mode}/ag/{method}/{evo_setting}/time_-1/query_{query_id}_n_{n_stages}"
            data_path = f"{save_data_header}/query_control_{is_query_control}/latest_model_{cpu_mode}/{model_name}/oracle_{is_oracle}/{method}/{evo_setting}/time_-1/query_{query_id}_n_{n_stages}"

            F = np.loadtxt(f"{data_path}/F.txt")
            F_list.append(np.round(np.unique(F.reshape(-1, 2), axis=0), 5))
        elif method == "ws":
            # data_path = f"./output/output/test/latest_model_{cpu_mode}/ag/{method}/1000000_11/time_-1/query_{query_id}_n_{n_stages}"
            # data_path = f"./output/test/query_control_True/latest_model_{cpu_mode}/ag/{method}/{ws_setting}/time_-1/query_{query_id}_n_{n_stages}"
            data_path = f"{save_data_header}/query_control_{is_query_control}/latest_model_{cpu_mode}/{model_name}/oracle_{is_oracle}/{method}/{ws_setting}/time_-1/query_{query_id}_n_{n_stages}"

            F = np.loadtxt(f"{data_path}/F.txt")
            F_list.append(np.round(np.unique(F.reshape(-1, 2), axis=0), 5))
        elif method == "ppf":
            # data_path = f"./output/test/query_control_False/latest_model_{cpu_mode}/ag/ppf/1_1_1/time_-1/query_{query_id}_n_{n_stages}"

            # data_path = f"./output/test/query_control_True/latest_model_{cpu_mode}/ag/ppf/{ppf_setting}/time_-1/query_{query_id}_n_{n_stages}"
            data_path = f"{save_data_header}/query_control_{is_query_control}/latest_model_{cpu_mode}/{model_name}/oracle_{is_oracle}/{method}/{ppf_setting}/time_-1/query_{query_id}_n_{n_stages}"

            F = np.loadtxt(f"{data_path}/F.txt")
            F_list.append(np.round(np.unique(F.reshape(-1, 2), axis=0), 5))
        else:
            if sample_mode == "grid":
                for n_c_samples in n_c_samples_list:
                    for n_p_samples in n_p_samples_list:

                        # data_path = f"./output/test/drop_pygmo/latest_model_{cpu_mode}/ag/oracle_False/div_and_conq_moo%B/{n_c_samples}_{n_p_samples}/time_-1/query_{query_id}_n_{n_stages}/grid/B"

                        dag_opt_algo = method.split("%")[1]
                        div_moo_setting = f"{n_c_samples}_{n_p_samples}"
                        data_path = f"{save_data_header}/query_control_False/latest_model_{cpu_mode}/{model_name}/oracle_False/{method}/" \
                                    f"{div_moo_setting}/time_-1/query_{query_id}_n_{n_stages}/{sample_mode}/{dag_opt_algo}"

                        F = np.loadtxt(f"{data_path}/F.txt")
                        F_list.append(np.round(np.unique(F.reshape(-1, 2), axis=0), 5))
            else:
                for n_c_samples in n_c_samples_list:
                    for n_p_samples in n_p_samples_list:

                        # data_path = f"./output/test/drop_pygmo/latest_model_{cpu_mode}/ag/oracle_False/div_and_conq_moo%B/{n_c_samples}_{n_p_samples}/time_-1/query_{query_id}_n_{n_stages}/random/B"
                        dag_opt_algo = method.split("%")[1]
                        div_moo_setting = f"{n_c_samples}_{n_p_samples}"
                        data_path = f"{save_data_header}/query_control_False/latest_model_{cpu_mode}/{model_name}/oracle_False/{method}/" \
                                    f"{div_moo_setting}/time_-1/query_{query_id}_n_{n_stages}/{sample_mode}/{dag_opt_algo}"

                        F = np.loadtxt(f"{data_path}/F.txt")
                        F_list.append(np.round(np.unique(F.reshape(-1, 2), axis=0), 5))


    ## calculate the Nadir point among all methods
    nadir_lat, nadir_cost = -1, -1
    utopia_lat, utopia_cost = np.inf, np.inf
    for F in F_list:
        max_lat_F = max(np.array(F)[:, 0])
        max_cost_F = max(np.array(F)[:, 1])

        min_lat_F = min(np.array(F)[:, 0])
        min_cost_F = min(np.array(F)[:, 1])
        if max_lat_F > nadir_lat:
            nadir_lat = max_lat_F
        if max_cost_F > nadir_cost:
            nadir_cost = max_cost_F

        if min_lat_F < utopia_lat:
            utopia_lat = min_lat_F
        if min_cost_F < utopia_cost:
            utopia_cost = min_cost_F

    nadir = np.concatenate(F_list).max(axis=0)
    utopia = np.concatenate(F_list).min(axis=0)
    return nadir, utopia

def get_utopia_nadir_all_algos_w_query_control(methods, query_id, n_stages, sample_mode, cpu_mode="cpu",
                               n_c_samples_list=[-1], n_p_samples_list=[-1],
                               evo_setting="100_1000",
                               ws_n_samples_list = [],
                               ppf_setting="1_1_1",
                               save_data_header="",
                               is_query_control=False,
                               is_oracle=False):

    # new HV analaysis: to have consistent reference points for all algorithms and all settings
    F_list = []
    for method in methods:
        if method == "evo":
            # data_path = f"./output/output/test/latest_model_{cpu_mode}/ag/{method}/100_1000/time_-1/query_{query_id}_n_{n_stages}"
            # data_path = f"./output/test/query_control_True/latest_model_{cpu_mode}/ag/{method}/{evo_setting}/time_-1/query_{query_id}_n_{n_stages}"
            data_path = f"{save_data_header}/query_control_{is_query_control}/latest_model_{cpu_mode}/ag/oracle_{is_oracle}/{method}/{evo_setting}/time_-1/query_{query_id}_n_{n_stages}"

            F = np.loadtxt(f"{data_path}/F.txt")
            F_list.append(np.round(np.unique(F.reshape(-1, 2), axis=0), 5))
        elif method == "ws":
            # data_path = f"./output/output/test/latest_model_{cpu_mode}/ag/{method}/1000000_11/time_-1/query_{query_id}_n_{n_stages}"
            # data_path = f"./output/test/query_control_True/latest_model_{cpu_mode}/ag/{method}/{ws_setting}/time_-1/query_{query_id}_n_{n_stages}"
            for n_samples in ws_n_samples_list:
                ws_setting = f"{n_samples}_11"
                data_path = f"{save_data_header}/query_control_{is_query_control}/latest_model_{cpu_mode}/ag/oracle_{is_oracle}/{method}/{ws_setting}/time_-1/query_{query_id}_n_{n_stages}"

                F = np.loadtxt(f"{data_path}/F.txt")
                F_list.append(np.round(np.unique(F.reshape(-1, 2), axis=0), 5))
        elif method == "ppf":
            # data_path = f"./output/test/query_control_False/latest_model_{cpu_mode}/ag/ppf/1_1_1/time_-1/query_{query_id}_n_{n_stages}"

            # data_path = f"./output/test/query_control_True/latest_model_{cpu_mode}/ag/ppf/{ppf_setting}/time_-1/query_{query_id}_n_{n_stages}"
            data_path = f"{save_data_header}/query_control_{is_query_control}/latest_model_{cpu_mode}/ag/oracle_{is_oracle}/{method}/{ppf_setting}/time_-1/query_{query_id}_n_{n_stages}"

            F = np.loadtxt(f"{data_path}/F.txt")
            F_list.append(np.round(np.unique(F.reshape(-1, 2), axis=0), 5))
        else:
            if sample_mode == "grid":
                for n_c_samples in n_c_samples_list:
                    for n_p_samples in n_p_samples_list:

                        # data_path = f"./output/test/drop_pygmo/latest_model_{cpu_mode}/ag/oracle_False/div_and_conq_moo%B/{n_c_samples}_{n_p_samples}/time_-1/query_{query_id}_n_{n_stages}/grid/B"

                        dag_opt_algo = method.split("%")[1]
                        div_moo_setting = f"{n_c_samples}_{n_p_samples}"
                        data_path = f"{save_data_header}/query_control_False/latest_model_{cpu_mode}/ag/oracle_False/{method}/" \
                                    f"{div_moo_setting}/time_-1/query_{query_id}_n_{n_stages}/{sample_mode}/{dag_opt_algo}"

                        F = np.loadtxt(f"{data_path}/F.txt")
                        F_list.append(np.round(np.unique(F.reshape(-1, 2), axis=0), 5))
            else:
                for n_c_samples in n_c_samples_list:
                    for n_p_samples in n_p_samples_list:

                        # data_path = f"./output/test/drop_pygmo/latest_model_{cpu_mode}/ag/oracle_False/div_and_conq_moo%B/{n_c_samples}_{n_p_samples}/time_-1/query_{query_id}_n_{n_stages}/random/B"
                        dag_opt_algo = method.split("%")[1]
                        div_moo_setting = f"{n_c_samples}_{n_p_samples}"
                        data_path = f"{save_data_header}/query_control_False/latest_model_{cpu_mode}/ag/oracle_False/{method}/" \
                                    f"{div_moo_setting}/time_-1/query_{query_id}_n_{n_stages}/{sample_mode}/{dag_opt_algo}"

                        F = np.loadtxt(f"{data_path}/F.txt")
                        F_list.append(np.round(np.unique(F.reshape(-1, 2), axis=0), 5))


    ## calculate the Nadir point among all methods
    nadir_lat, nadir_cost = -1, -1
    utopia_lat, utopia_cost = np.inf, np.inf
    for F in F_list:
        max_lat_F = max(np.array(F)[:, 0])
        max_cost_F = max(np.array(F)[:, 1])

        min_lat_F = min(np.array(F)[:, 0])
        min_cost_F = min(np.array(F)[:, 1])
        if max_lat_F > nadir_lat:
            nadir_lat = max_lat_F
        if max_cost_F > nadir_cost:
            nadir_cost = max_cost_F

        if min_lat_F < utopia_lat:
            utopia_lat = min_lat_F
        if min_cost_F < utopia_cost:
            utopia_cost = min_cost_F

    nadir = np.concatenate(F_list).max(axis=0)
    utopia = np.concatenate(F_list).min(axis=0)
    return nadir, utopia

def plot_err_bar(
    mean_list,
    std_list,
    methods,
    queries,
    sample_mode,
    mode="time",
    cpu_mode="cpu",
    plot_mode="comp_all",
    temp="tpch",
    is_query_control=False,
    save_fig=False,
    save_data_header="",
    expr="expr",
):
    # short_methods = ["Div_MOO(random)", "Div_MOO(grid)", "WS", "EVO"]
    X = np.arange(len(methods))
    fig2, ax2 = plt.subplots()

    if mode == "time":
        ax2.set_yscale("log")
        ax2.bar(X, mean_list, yerr=std_list, color="blue", ecolor="gray", capsize=5)
        ax2.set_ylabel("Time cost (s) among all queries")
        for a, b in zip(X + 0.00, mean_list):
            ax2.text(
                a,
                b,
                "%.3f" % b,
                ha="center",
                va="bottom",
            )
        ax2.plot(X, np.ones_like(X), ls='--', c='red')
    elif mode == "hv":
        ax2.set_ylim([0, 100])
        ax2.bar(X, mean_list, yerr=std_list, color="cyan", ecolor="gray", capsize=5)
        ax2.set_ylabel("HyperVolume (%) among all queries")
        for a, b in zip(X + 0.00, mean_list):
            ax2.text(
                a,
                b,
                "%.3f" % b,
                ha="center",
                va="bottom",
                fontsize = 8
            )
    else:
        raise Exception(f"mode {mode} is not supported!")

    ax2.set_xlabel("Query-level MOO")
    ax2.set_xticks(X, methods)
    # ax2.set_xticks(X, short_methods)
    ax2.legend()
    if isinstance(queries, str):
        ax2.set_title(f"{queries}")
    elif isinstance(queries, list):
        # ax2.set_title(f"All {temp} queries with initial theta ({sample_mode}, {cpu_mode}, query_control_{is_query_control})")
        ax2.set_title(
            f"All {temp} queries")

    else:
        raise Exception(f"queries format {queries} is not supported!")

    if "dag_opt" in plot_mode:
        pass
    else:
        # plt.xticks(rotation=90)
        pass
    plt.tight_layout()
    # plt.show()

    if save_fig:
        plot_path = f"{save_data_header}/plots/{expr}/HV/query_control_{is_query_control}"
        if not os.path.exists(plot_path):
            os.makedirs(plot_path, exist_ok=True)
        plt.savefig(f"{plot_path}/{temp}.pdf")


def plot_box(X, methods, sample_mode, cpu_mode, plot_mode="comp_all", temp="tpch",
             is_query_control=False, is_save_fig=False,
             save_data_header="",
             expr="expr"):
    x = np.array(X).T

    fig, ax = plt.subplots()
    ax.set_yscale('log')
    # ax.set_ylim([0, 5])
    ax.boxplot(x.tolist(), labels=methods)
    ax.plot(np.arange(1, len(methods) + 1), np.ones_like(np.arange(len(methods))), ls='--', c='red')

    ax.set_xlabel("Query-level MOO")
    ax.set_ylabel("Time cost (s)")
    # ax.set_xticks(x, methods)
    if "dag_opt" in plot_mode:
        pass
    else:
        # plt.xticks(rotation=90)
        pass
    # ax.set_title(f"All {temp} queries for initial theta ({sample_mode}, {cpu_mode}, query_control_{is_query_control})")
    ax.set_title(
        f"All {temp} queries")
    plt.tight_layout()
    # plt.show()
    if is_save_fig:
        plot_path = f"{save_data_header}/plots/{expr}/time/query_control_{is_query_control}"
        if not os.path.exists(plot_path):
            os.makedirs(plot_path, exist_ok=True)
        plt.savefig(f"{plot_path}/{temp}.pdf")

def pareto_frontier(queries, stages, cpu_mode, sample_mode, temp):

    # methods = ["ws", "evo", "div_and_conq_moo%B%grid", "div_and_conq_moo%B%random"]
    methods = ["ws", "evo", "ppf", "div_and_conq_moo%B%grid"]
    algo_labels = []
    markers = []
    colors = []
    for query_id, n_stages in zip(queries, stages):
        F_list = []
        for method in methods:

            if "div_and_conq_moo" in method:
                if "grid" in method:
                    # color_list = ["orangered", "lightsalmon", "indianred", "brown"]
                    # marker_list = ["1", "3", "x", "v"]
                    x=np.arange(30)
                    for n_c_samples in [16, 32, 64, 128, 256]:
                        for n_p_samples in [16, 32, 64, 128, 256, 512]:
                            data_path = f"./output/test/drop_pygmo/latest_model_{cpu_mode}/ag/oracle_False/div_and_conq_moo%B/{n_c_samples}_{n_p_samples}/time_-1/query_{query_id}_n_{n_stages}/grid/B"
                            F = np.loadtxt(f"{data_path}/F.txt")
                            F_list.append(F)
                            algo_labels.append(f"grid_c_{n_c_samples}_p{n_p_samples}")
                    # algo_labels.append("div_and_conq_moo_grid")
                    # markers.append("+")
                    # colors.append("green")
                    colors.extend(x.tolist())
                    # markers.extend(marker_list)
                else:
                    color_list = ["mistyrose", "salmon", "tomato", "darksalmon", "coral",
                                  "orangered", "lightsalmon", "indianred", "brown"]
                    marker_list = ["1", "3", "x", "v", "p", "^", "<", "*", "."]
                    for n_c_samples in [64, 128, 256]:
                        for n_p_samples in [64, 128, 512]:
                            data_path = f"./output/test/latest_model_{cpu_mode}/ag/div_and_conq_moo%B/{n_c_samples}_{n_p_samples}/time_-1/query_{query_id}_n_{n_stages}/random/B"
                            F = np.loadtxt(f"{data_path}/F.txt")
                            F_list.append(F)
                            algo_labels.append(f"random_c_{n_c_samples}_p_{n_p_samples}")
                            # markers.append("1")
                    colors.extend(color_list)
                    markers.extend(marker_list)
            elif method == "evo":
                data_path = f"./output/output/test/latest_model_{cpu_mode}/ag/{method}/100_1000/time_-1/query_{query_id}_n_{n_stages}"
                F = np.loadtxt(f"{data_path}/F.txt")
                F_list.append(F)
                algo_labels.append(method)
                markers.append("o")
                colors.append("blue")
            elif method == "ws":
                data_path = f"./output/output/test/latest_model_{cpu_mode}/ag/{method}/1000000_11/time_-1/query_{query_id}_n_{n_stages}"
                F = np.loadtxt(f"{data_path}/F.txt")
                F_list.append(F)
                algo_labels.append(method)
                markers.append("s")
                colors.append("orange")
            elif method == "ppf":
                data_path = f"./output/test/query_control_False/latest_model_{cpu_mode}/ag/{method}/1_2_2/time_-1/query_{query_id}_n_{n_stages}"
                F = np.loadtxt(f"{data_path}/F.txt")
                F_list.append(F)
                algo_labels.append(method)
                markers.append("*")
                colors.append("rose")
            else:
                raise Exception(f"Method {method} is not supported!")

        fig, ax = plt.subplots()

        # colors = np.arange(len(algo_labels))
        colors_auto = cm.rainbow(np.linspace(0, 1, len(algo_labels)))
        # for F, m, marker, color in zip(F_list, algo_labels, markers, colors):
        for F, m, color in zip(F_list, algo_labels, colors_auto):
            if len(F) > 0:
                q_lat = np.array(F).reshape(-1, 2)[:, 0]
                q_cost = np.array(F).reshape(-1, 2)[:, 1]

                # ax.scatter(
                #     q_cost, q_lat, marker=marker, alpha=0.6, c=np.ones_like(q_cost) * color, label=f"{m} ({len(F)})"
                # )
                ax.scatter(
                    q_cost, q_lat, alpha=0.6, color=color, label=f"{m} ({len(F)})"
                )
            else:
                ax.scatter(F, F, color=color, label=f"{m} ({len(F)})")
        ax.set_xlabel("Query cost")
        ax.set_ylabel("Query latency")
        ax.set_title(f"{temp} {query_id}_n_subQ_{n_stages} ({cpu_mode})")
        ax.legend()
        # plt.show()
        file_path = f"./output/plots/grids/all/"
        if not os.path.exists(file_path):
            os.makedirs(file_path, exist_ok=True)
        plt.savefig(f"{file_path}/query_{query_id}_n_subQ_{n_stages}.pdf")


def pareto_frontier_all_fine_control(queries, stages, cpu_mode, sample_mode, div_moo_setting,
                          evo_setting, ws_setting, ppf_setting, save_data_header, is_query_control, is_oracle, model_name, temp):

    # methods = ["ws", "evo", "div_and_conq_moo%B%grid", "div_and_conq_moo%B%random"]
    methods = ["ws", "evo", "ppf", "div_and_conq_moo%B"]
    algo_labels = []
    markers = []
    colors = []
    for query_id, n_s in zip(queries, stages):
        F_list = []
        for method in methods:
            if "div_and_conq_moo" in method:
                dag_opt_algo = method.split("%")[1]
                # div_moo_setting = f"{n_c_samples}_{n_p_samples}"
                data_path = f"{save_data_header}/query_control_False/latest_model_{cpu_mode}/{model_name}/oracle_False/{method}/" \
                            f"{div_moo_setting}/time_-1/query_{query_id}_n_{n_s}/{sample_mode}/{dag_opt_algo}"
                F = np.loadtxt(f"{data_path}/F.txt")
                F_list.append(F)
            elif method == "evo":
                # data_path = f"./output/output/test/latest_model_{cpu_mode}/ag/{method}/100_1000/time_-1/query_{query_id}_n_{n_stages}"
                data_path = f"{save_data_header}/query_control_{is_query_control}/latest_model_{cpu_mode}/{model_name}/oracle_{is_oracle}/{method}/{evo_setting}/time_-1/query_{query_id}_n_{n_s}"

                F = np.loadtxt(f"{data_path}/F.txt")
                F_list.append(F)
            elif method == "ws":
                # data_path = f"./output/output/test/latest_model_{cpu_mode}/ag/{method}/1000000_11/time_-1/query_{query_id}_n_{n_stages}"
                data_path = f"{save_data_header}/query_control_{is_query_control}/latest_model_{cpu_mode}/{model_name}/oracle_{is_oracle}/{method}/{ws_setting}/time_-1/query_{query_id}_n_{n_s}"
                F = np.loadtxt(f"{data_path}/F.txt")
                F_list.append(F)
            elif method == "ppf":
                # data_path = f"./output/test/query_control_False/latest_model_{cpu_mode}/ag/{method}/1_2_2/time_-1/query_{query_id}_n_{n_stages}"
                data_path = f"{save_data_header}/query_control_{is_query_control}/latest_model_{cpu_mode}/{model_name}/oracle_{is_oracle}/{method}/{ppf_setting}/time_-1/query_{query_id}_n_{n_s}"

                F = np.loadtxt(f"{data_path}/F.txt")
                F_list.append(F)
            else:
                raise Exception(f"Method {method} is not supported!")

        fig, ax = plt.subplots()

        # colors = np.arange(len(algo_labels))
        colors_auto = cm.rainbow(np.linspace(0, 1, len(methods)))
        # for F, m, marker, color in zip(F_list, algo_labels, markers, colors):
        for F, m, color in zip(F_list, methods, colors_auto):
            if len(F) > 0:
                q_lat = np.array(F).reshape(-1, 2)[:, 0]
                q_cost = np.array(F).reshape(-1, 2)[:, 1]

                # ax.scatter(
                #     q_cost, q_lat, marker=marker, alpha=0.6, c=np.ones_like(q_cost) * color, label=f"{m} ({len(F)})"
                # )
                ax.scatter(
                    q_cost, q_lat, alpha=0.6, color=color, label=f"{m} ({len(F)})"
                )
            else:
                ax.scatter(F, F, color=color, label=f"{m} ({len(F)})")
        ax.set_xlabel("Query cost")
        ax.set_ylabel("Query latency")
        ax.set_title(f"{temp} {query_id}_n_subQ_{n_s} ({cpu_mode})")
        ax.legend()
        # plt.show()
        file_path = f"{save_data_header}/plots/compare_moo_algos_fine_control/"
        if not os.path.exists(file_path):
            os.makedirs(file_path, exist_ok=True)
        plt.savefig(f"{file_path}/query_{query_id}_n_subQ_{n_s}.pdf")

def comp_pareto_frontier_query_fine_control(queries, stages, cpu_mode, sample_mode, div_moo_setting,
                          evo_setting, ws_setting, ppf_setting, save_data_header, is_oracle, model_name, temp):

    methods = ["ws", "evo", "ppf", "div_and_conq_moo%B"]
    for query_id, n_s in zip(queries, stages):
        F_fine_list = []
        F_query_list = []
        for method in methods:

            if "div_and_conq_moo" in method:
                dag_opt_algo = method.split("%")[1]
                data_path = f"{save_data_header}/query_control_False/latest_model_{cpu_mode}/{model_name}/oracle_False/{method}/" \
                            f"{div_moo_setting}/time_-1/query_{query_id}_n_{n_s}/{sample_mode}/{dag_opt_algo}"
                F_fine = np.loadtxt(f"{data_path}/F.txt")
                F_query = []

            elif method == "evo":
                # data_path = f"./output/output/test/latest_model_{cpu_mode}/ag/{method}/100_1000/time_-1/query_{query_id}_n_{n_stages}"
                data_path_fine = f"{save_data_header}/query_control_False/latest_model_{cpu_mode}/{model_name}/oracle_{is_oracle}/{method}/{evo_setting}/time_-1/query_{query_id}_n_{n_s}"
                data_path_query = f"{save_data_header}/query_control_True/latest_model_{cpu_mode}/{model_name}/oracle_{is_oracle}/{method}/{evo_setting}/time_-1/query_{query_id}_n_{n_s}"
                F_fine = np.loadtxt(f"{data_path_fine}/F.txt")
                F_query = np.loadtxt(f"{data_path_query}/F.txt")

            elif method == "ws":
                # data_path = f"./output/output/test/latest_model_{cpu_mode}/ag/{method}/1000000_11/time_-1/query_{query_id}_n_{n_stages}"
                data_path_fine = f"{save_data_header}/query_control_False/latest_model_{cpu_mode}/{model_name}/oracle_{is_oracle}/{method}/{ws_setting}/time_-1/query_{query_id}_n_{n_s}"
                data_path_query = f"{save_data_header}/query_control_True/latest_model_{cpu_mode}/{model_name}/oracle_{is_oracle}/{method}/{ws_setting}/time_-1/query_{query_id}_n_{n_s}"
                F_fine = np.loadtxt(f"{data_path_fine}/F.txt")
                F_query = np.loadtxt(f"{data_path_query}/F.txt")

            elif method == "ppf":
                # data_path = f"./output/test/query_control_False/latest_model_{cpu_mode}/ag/{method}/1_2_2/time_-1/query_{query_id}_n_{n_stages}"
                data_path_fine = f"{save_data_header}/query_control_False/latest_model_{cpu_mode}/{model_name}/oracle_{is_oracle}/{method}/{ppf_setting}/time_-1/query_{query_id}_n_{n_s}"
                data_path_query = f"{save_data_header}/query_control_True/latest_model_{cpu_mode}/{model_name}/oracle_{is_oracle}/{method}/{ppf_setting}/time_-1/query_{query_id}_n_{n_s}"
                F_fine = np.loadtxt(f"{data_path_fine}/F.txt")
                F_query = np.loadtxt(f"{data_path_query}/F.txt")

            else:
                raise Exception(f"Method {method} is not supported!")

            F_fine_list.append(F_fine)
            F_query_list.append(F_query)

        fig, ax = plt.subplots()

        # colors = np.arange(len(algo_labels))
        colors_auto = cm.rainbow(np.linspace(0, 1, len(methods)))
        # for F, m, marker, color in zip(F_list, algo_labels, markers, colors):
        for F_fine, F_query, m, color in zip(F_fine_list, F_query_list, methods, colors_auto):
            if len(F_fine) > 0:
                q_lat_fine = np.array(F_fine).reshape(-1, 2)[:, 0]
                q_cost_fine = np.array(F_fine).reshape(-1, 2)[:, 1]

                ax.scatter(
                    q_cost_fine, q_lat_fine, alpha=0.6, marker="o", color=color, label=f"{m}_subQ ({len(F_fine)})"
                )

                if len(F_query) > 0:
                    q_lat_query = np.array(F_query).reshape(-1, 2)[:, 0]
                    q_cost_query = np.array(F_query).reshape(-1, 2)[:, 1]

                    ax.scatter(
                        q_cost_query, q_lat_query, alpha=0.6, marker="+", color=color, label=f"{m}_query ({len(F_query)})"
                    )

            else:
                ax.scatter(F_fine, F_fine, color=color, label=f"{m} ({len(F_fine)})")
        ax.set_xlabel("Query cost")
        ax.set_ylabel("Query latency")
        ax.set_title(f"{temp} {query_id}_n_subQ_{n_s} ({cpu_mode})")
        ax.legend()
        # plt.show()
        file_path = f"{save_data_header}/plots/compare_fine_query_control"
        if not os.path.exists(file_path):
            os.makedirs(file_path, exist_ok=True)
        plt.savefig(f"{file_path}/query_{query_id}_n_subQ_{n_s}.pdf")

def save_utopia_and_nadir(queries, stages, cpu_mode, sample_mode, n_c_samples_list, n_p_samples_list,
                          evo_setting, ws_setting, ppf_setting, save_data_header, is_query_control, is_oracle, model_name):
    methods = ["ws", "evo", "ppf", "div_and_conq_moo%B"]
    for i, (query_id, n_s) in enumerate(zip(queries, stages)):
        all_nadir, all_utopia = get_utopia_nadir_all_algos(methods, query_id, n_s, sample_mode, cpu_mode,
                                                           n_c_samples_list,
                                                           n_p_samples_list,
                                                           evo_setting,
                                                           ws_setting,
                                                           ppf_setting,
                                                           save_data_header=save_data_header,
                                                           is_query_control=is_query_control,
                                                           is_oracle=is_oracle,
                                                           model_name=model_name,
                                                           )
        print(f"the utopia in the query {query_id} is:")
        print(all_utopia)
        print(f"the nadir in the query {query_id} is:")
        print(all_nadir)
        print()
        results = np.vstack((all_utopia, all_nadir))
        save_path = f"{save_data_header}/utopia_and_nadir/query_{query_id}_n_{n_s}"
        save_results(save_path, results=results, mode="utopia_and_nadir")

def compare_query_finer_control_HV(queries, stages, cpu_mode, sample_mode, evo_setting, ws_setting, ppf_setting,
                                   div_moo_setting,
                                   save_data_header="",
                                   is_query_control=False, is_oracle=False,
                                   model_name="mlp", temp="tpch", expr="expr", save_fig=False):
    markers = ["s", "o", "*", "+"]
    colors = ["blue", "orange", "green", "cyan"]
    # methods = ["ws", "evo", "ppf", "approx_solve"]
    methods = ["ws", "evo", "ppf", "div_and_conq_moo%B"]
    # methods = ["div_and_conq_moo%B"]
    algo_labels = []

    all_F_list = []
    all_hv_list, all_tc_list = [], []
    for i, (query_id, n_s) in enumerate(zip(queries, stages)):
        utopia_and_nadir_path = f"{save_data_header}/utopia_and_nadir/query_{query_id}_n_{n_s}"

        # utopia_and_nadir_path = f"./output/0218test/utopia_and_nadir/query_{query_id}_n_{n_s}"

        utopia_and_nadir = np.loadtxt(f"{utopia_and_nadir_path}/utopia_and_nadir.txt")
        all_utopia = utopia_and_nadir[0, :]
        all_nadir = utopia_and_nadir[1, :]

        F_list = []
        hv_list, tc_list = [], []
        for method in methods:
            if method == "evo":
                data_path = f"{save_data_header}/query_control_{is_query_control}/latest_model_{cpu_mode}/{model_name}/oracle_{is_oracle}/{method}/{evo_setting}/time_-1/query_{query_id}_n_{n_s}"
                F = np.loadtxt(f"{data_path}/F.txt")

                F_list.append(np.around(np.unique(F.reshape(-1, 2), axis=0), 5))

                tc_dict_file = f"{data_path}/end_to_end.json"
                with open(tc_dict_file) as f:
                    d = json.load(f)
                tc = d["end_to_end"]
                tc_list.append(tc)
                if i == 0:
                    algo_labels.append(method)
            elif method == "ws":
                data_path = f"{save_data_header}/query_control_{is_query_control}/latest_model_{cpu_mode}/{model_name}/oracle_{is_oracle}/{method}/{ws_setting}/time_-1/query_{query_id}_n_{n_s}"
                F = np.loadtxt(f"{data_path}/F.txt")

                F_list.append(np.around(np.unique(F.reshape(-1, 2), axis=0), 5))

                tc_dict_file = f"{data_path}/end_to_end.json"
                with open(tc_dict_file) as f:
                    d = json.load(f)
                tc = d["end_to_end"]
                tc_list.append(tc)
                if i == 0:
                    algo_labels.append(method)
            elif method == "ppf":
                # if is_query_control:
                #     data_path = f"./output/test/query_control_True/latest_model_{cpu_mode}/ag/ppf/{ppf_setting}/time_-1/query_{query_id}_n_{n_s}"
                # else:
                #     data_path = f"./output/test/query_control_False/latest_model_{cpu_mode}/ag/ppf/{ppf_setting}/time_-1/query_{query_id}_n_{n_s}"
                data_path = f"{save_data_header}/query_control_{is_query_control}/latest_model_{cpu_mode}/{model_name}/oracle_{is_oracle}/{method}/{ppf_setting}/time_-1/query_{query_id}_n_{n_s}"

                F = np.loadtxt(f"{data_path}/F.txt")

                F_list.append(np.around(np.unique(F.reshape(-1, 2), axis=0), 5))
                # tc = np.loadtxt(f"{data_path}/time.txt")
                # tc_list.append(tc.item())
                tc_dict_file = f"{data_path}/end_to_end.json"
                with open(tc_dict_file) as f:
                    d = json.load(f)
                tc = d["end_to_end"]
                tc_list.append(tc)
                if i == 0:
                    algo_labels.append(method)
            else:

                dag_opt_algo = method.split("%")[1]
                data_path = f"{save_data_header}/query_control_False/latest_model_{cpu_mode}/{model_name}/oracle_False/{method}/" \
                            f"{div_moo_setting}/time_-1/query_{query_id}_n_{n_s}/{sample_mode}/{dag_opt_algo}"
                F = np.loadtxt(f"{data_path}/F.txt")

                F_list.append(np.around(np.unique(F.reshape(-1, 2), axis=0), 5))
                # tc = np.loadtxt(f"{data_path}/time.txt")
                # tc_list.append(tc.item())
                tc_dict_file = f"{data_path}/end_to_end.json"
                with open(tc_dict_file) as f:
                    d = json.load(f)
                tc = d["end_to_end"]
                tc_list.append(tc)
                if i == 0:
                    algo_labels.append(f"ours")

                #
                hv = cal_hv(F_list, all_nadir, all_utopia)

                if max(hv) > 100:
                    print("There is something wrong with the reference points!")

                hv_list.extend(hv)

        all_hv_list.append(hv_list)
        all_tc_list.append(tc_list)
        all_F_list.append(F_list)

    hv_mean = np.mean(np.array(all_hv_list), axis=0)
    tc_mean = np.mean(np.array(all_tc_list).reshape(-1, hv_mean.shape[0]), axis=0)

    hv_std = np.std(np.array(all_hv_list), axis=0)
    tc_std = np.std(np.array(all_tc_list), axis=0)

    hv_min = np.min(np.array(all_hv_list), axis=0)
    tc_min = np.min(np.array(all_tc_list), axis=0)

    hv_max = np.max(np.array(all_hv_list), axis=0)
    tc_max = np.max(np.array(all_tc_list), axis=0)

    print(f"mean HV with mode {cpu_mode} is {hv_mean}")
    print(f"std HV with mode {cpu_mode} is {hv_std}")
    print(f"min HV with mode {cpu_mode} is {hv_min}")
    print(f"max HV with mode {cpu_mode} is {hv_max}")
    print(f"mean solving time with mode {cpu_mode} is {tc_mean}")
    print(f"std solving time with mode {cpu_mode} is {tc_std}")
    print(f"min solving time with mode {cpu_mode} is {tc_min}")
    print(f"max solving time with mode {cpu_mode} is {tc_max}")
    print()
    our_vs_ws_hv = (hv_mean[-1] - hv_mean[0]) / hv_mean[0]
    our_vs_evo_hv = (hv_mean[-1] - hv_mean[1]) / hv_mean[1]
    our_vs_ppf_hv = (hv_mean[-1] - hv_mean[2]) / hv_mean[2]
    print(f"the improvement of HV v.s. (ws, evo, ppf) is: "
          f"{our_vs_ws_hv}, {our_vs_evo_hv}, {our_vs_ppf_hv}")

    our_vs_ws_tc = (tc_mean[0] - tc_mean[-1]) / tc_mean[0]
    our_vs_evo_tc = (tc_mean[1] - tc_mean[-1]) / tc_mean[1]
    our_vs_ppf_tc = (tc_mean[2] - tc_mean[-1]) / tc_mean[2]
    print(f"the improvement of Time cost v.s. (ws, evo, ppf) is: "
          f"{our_vs_ws_tc}, {our_vs_evo_tc}, {our_vs_ppf_tc}")

    plot_err_bar(hv_mean, hv_std, algo_labels, queries, sample_mode, mode="hv", cpu_mode=cpu_mode, temp=temp,
                 is_query_control=is_query_control,
                 save_data_header=save_data_header,
                 expr=expr, save_fig=save_fig)
    # plot_err_bar(tc_mean, tc_std, algo_labels, queries, sample_mode, mode="time", cpu_mode=cpu_mode)
    plot_box(all_tc_list, algo_labels, sample_mode, cpu_mode=cpu_mode, temp=temp, is_query_control=is_query_control,
             save_data_header=save_data_header, expr=expr, is_save_fig=save_fig)

def compare_diff_dag_opt(queries, stages, cpu_mode, sample_mode, div_moo_setting, save_data_header="",
                                   is_query_control=False, is_oracle=False, save_pareto_front=False,
                         model_name="mlp", save_fig=False, expr="expr", temp="tpch"):
    methods = ["div_and_conq_moo%GD", "div_and_conq_moo%WS&11", "div_and_conq_moo%B"]


    all_tc_list = []
    all_hv_list = []

    for query_id, n_s in zip(queries, stages):
        # utopia_and_nadir_path = f"{save_data_header}/utopia_and_nadir/query_{query_id}_n_{n_s}"
        # utopia_and_nadir = np.loadtxt(f"{utopia_and_nadir_path}/utopia_and_nadir.txt")
        # all_utopia = utopia_and_nadir[0, :]
        # all_nadir = utopia_and_nadir[1, :]
        F_list = []
        hv_list = []
        tc_list = []
        algo_labels = []
        utopia_and_nadir_path = f"{save_data_header}/utopia_and_nadir/query_{query_id}_n_{n_s}"
        utopia_and_nadir = np.loadtxt(f"{utopia_and_nadir_path}/utopia_and_nadir.txt")
        all_utopia = utopia_and_nadir[0, :]
        all_nadir = utopia_and_nadir[1, :]

        if save_pareto_front:
            fig, ax = plt.subplots()
            colors = ["blue", "orange", "green"]
            markers = ["s", "o", "*"]

        for i, method in enumerate(methods):
            dag_opt_algo = method.split("%")[1]
            # div_moo_setting = f"{n_c_samples}_{n_p_samples}"

            if "WS" in dag_opt_algo:
                # data_path = f"./output/0218test/norm_div_WS/query_control_False/latest_model_{cpu_mode}/ag/oracle_False/{method}/" \
                #             f"{div_moo_setting}/time_-1/query_{query_id}_n_{n_s}/{sample_mode}/{dag_opt_algo}"
                data_path = f"{save_data_header}/query_control_False/latest_model_{cpu_mode}/{model_name}/oracle_False/{method}/" \
                            f"{div_moo_setting}/time_-1/query_{query_id}_n_{n_s}/{sample_mode}/{dag_opt_algo}"
                algo_labels.append("WS-based")
            else:
                data_path = f"{save_data_header}/query_control_False/latest_model_{cpu_mode}/{model_name}/oracle_False/{method}/" \
                        f"{div_moo_setting}/time_-1/query_{query_id}_n_{n_s}/{sample_mode}/{dag_opt_algo}"
                algo_labels.append(dag_opt_algo)

            F = np.loadtxt(f"{data_path}/F.txt")
            F_uniq = np.around(np.unique(F.reshape(-1, 2), axis=0), 5)
            F_list.append(F_uniq)
            # tc = np.loadtxt(f"{data_path}/time.txt")
            # tc_list.append(tc.item())
            tc_dict_file = f"{data_path}/end_to_end.json"
            with open(tc_dict_file) as f:
                d = json.load(f)
            tc = d["end_to_end"]
            tc_list.append(tc)

            if save_pareto_front:
                ax.scatter(F[:, 1], F[:, 0], marker=markers[i], color=colors[i], label=f"{algo_labels[i]} ({F_uniq.shape[0]})")
                ax.legend()
                ax.set_xlabel("Query Cost")
                ax.set_ylabel("Query Latency")
                ax.set_title(f"Pareto frontier of query {query_id} with {n_s} subQs")
                # plt.tight_layout()
                # plt.show()
                plot_path = f"{save_data_header}/plots/{expr}/compare_dag_opt"
                if not os.path.exists(plot_path):
                    os.makedirs(plot_path, exist_ok=True)
                plt.savefig(f"{plot_path}/query_{query_id}_n_subQ_{n_s}.pdf")
        hv = cal_hv(F_list, all_nadir, all_utopia)

        all_tc_list.append(tc_list)

        if max(hv) > 100:
            print("There is something wrong with the reference points!")

        all_hv_list.append(hv)

    hv_mean = np.mean(np.array(all_hv_list), axis=0)
    tc_mean = np.mean(np.array(all_tc_list), axis=0)

    hv_std = np.std(np.array(all_hv_list), axis=0)
    tc_std = np.std(np.array(all_tc_list), axis=0)

    hv_min = np.min(np.array(all_hv_list), axis=0)
    tc_min = np.min(np.array(all_tc_list), axis=0)

    hv_max = np.max(np.array(all_hv_list), axis=0)
    tc_max = np.max(np.array(all_tc_list), axis=0)

    print(f"mean HV with mode {cpu_mode} is {hv_mean}")
    print(f"std HV with mode {cpu_mode} is {hv_std}")
    print(f"min HV with mode {cpu_mode} is {hv_min}")
    print(f"max HV with mode {cpu_mode} is {hv_max}")
    print(f"mean solving time with mode {cpu_mode} is {tc_mean}")
    print(f"std solving time with mode {cpu_mode} is {tc_std}")
    print(f"min solving time with mode {cpu_mode} is {tc_min}")
    print(f"max solving time with mode {cpu_mode} is {tc_max}")
    print()

    plot_err_bar(hv_mean, hv_std, algo_labels, queries, sample_mode, mode="hv", cpu_mode=cpu_mode,
                 plot_mode="comp_dag_opt", save_fig=save_fig,
                 save_data_header=save_data_header,
                 expr=expr, temp=temp)
    # plot_err_bar(tc_mean, tc_std, algo_labels, queries, sample_mode, mode="time", cpu_mode=cpu_mode)
    plot_box(all_tc_list, methods, algo_labels, cpu_mode=cpu_mode,
             plot_mode="comp_dag_opt", is_save_fig=save_fig,
             save_data_header=save_data_header,
             expr=expr, temp=temp)

def compare_fine_query_control_diff_setting(queries, stages, cpu_mode, save_data_header, ws_n_samples_list = [],
                                            is_oracle=False, save_fig=False, temp="tpch", expr="expr"):

    all_hv_fine_list = []
    all_hv_query_list = []
    # n_samples_list = [10_000, 20_000, 50_000, 100_000, 10_000_000]
    algo_labels = [str(x) for x in ws_n_samples_list]

    F_fine_list = []
    F_query_list = []
    for n_samples in ws_n_samples_list:

        method = "ws"
        hv_fine_list = []
        hv_query_list = []
        for query_id, n_s in zip(queries, stages):
            utopia_and_nadir_path = f"{save_data_header}/utopia_and_nadir_w_query_control/query_{query_id}_n_{n_s}"
            utopia_and_nadir = np.loadtxt(f"{utopia_and_nadir_path}/utopia_and_nadir.txt")
            utopia = utopia_and_nadir[0, :]
            nadir = utopia_and_nadir[1, :]

            ws_setting = f"{n_samples}_11"
            data_path_fine = f"{save_data_header}/query_control_False/latest_model_{cpu_mode}/ag/oracle_{is_oracle}/{method}/{ws_setting}/time_-1/query_{query_id}_n_{n_s}"
            data_path_query = f"{save_data_header}/query_control_True/latest_model_{cpu_mode}/ag/oracle_{is_oracle}/{method}/{ws_setting}/time_-1/query_{query_id}_n_{n_s}"

            F_fine = np.loadtxt(f"{data_path_fine}/F.txt")
            F_query = np.loadtxt(f"{data_path_query}/F.txt")

            F_fine_list.append(F_fine)
            F_query_list.append(F_query)

            hv = cal_hv([F_fine, F_query], nadir, utopia)
            hv_fine_list.append(hv[0])
            hv_query_list.append(hv[1])

            if max(hv) > 100:
                print("There is something wrong with the reference points!")

            # if "2-1" in query_id:
            #     fig, ax = plt.subplots()
            #     ax.scatter(F_fine[:, 1], F_fine[:, 0], marker="o", color="blue", label="SubQ")
            #     ax.scatter(F_query[:, 1], F_query[:, 0], marker="x", color="orange", label="Query")
            #     ax.set_xlabel("Query Cost")
            #     ax.set_ylabel("Query Latency")
            #     ax.set_title(f"Query {query_id} with {n_s} subQs")
            #     ax.legend()
            #     plt.tight_layout()
            #     plt.show()


        all_hv_fine_list.append(hv_fine_list)
        all_hv_query_list.append(hv_query_list)

    hv_mean_fine = np.mean(np.array(all_hv_fine_list), axis=1)
    hv_std_fine = np.std(np.array(all_hv_fine_list), axis=1)

    hv_mean_query = np.mean(np.array(all_hv_query_list), axis=1)
    hv_std_query = np.std(np.array(all_hv_query_list), axis=1)

    print(f"FINER: mean HV with mode {cpu_mode} is {hv_mean_fine}")
    print(f"FINER: std HV with mode {cpu_mode} is {hv_std_fine}")
    print(f"QUERY: mean HV with mode {cpu_mode} is {hv_mean_query}")
    print(f"QUERY: std HV with mode {cpu_mode} is {hv_std_query}")

    # plot_err_bar(hv_mean_fine, hv_std_fine, algo_labels, queries, sample_mode="grid", mode="hv", cpu_mode=cpu_mode)
    # plot_err_bar(hv_mean_query, hv_std_query, algo_labels, queries, sample_mode="grid", mode="hv", cpu_mode=cpu_mode)
    plot_multi_bar_err(algo_labels, hv_mean_fine, hv_std_fine, hv_mean_query, hv_std_query,
                       save_fig=save_fig, expr=expr,
                       temp=temp)

def plot_multi_bar_err(algo_labels, hv_mean_fine, hv_std_fine, hv_mean_query, hv_std_query, save_fig, expr, temp):
    fig, ax = plt.subplots()
    width = 0.2
    x = np.arange(len(algo_labels))

    ax.set_ylim([0, 100])
    # ax.set_yscale("log")
    ax.bar(x, hv_mean_fine, width=width, yerr=hv_std_fine, color="cyan", ecolor="gray", capsize=5, label="SubQ")
    for a, b in zip(x + 0.00, hv_mean_fine):
        ax.text(
            a,
            b,
            "%.3f" % b,
            ha="center",
            va="bottom",
            fontsize=8
        )
    ax.bar(x + width, hv_mean_query, width=width, yerr=hv_std_query, color="orange", ecolor="gray", capsize=5,
           label="Query")
    for a, b in zip(x + width + 0.00, hv_mean_query):
        ax.text(
            a,
            b,
            "%.3f" % b,
            ha="center",
            va="bottom",
            fontsize=8
        )
    ax.set_ylabel("HyperVolume (%) among all queries")
    ax.set_xlabel("The number of samples in WS (n_ws=11)")
    ax.set_xticks(x + width)
    ax.set_xticklabels(algo_labels)
    ax.legend()
    plt.tight_layout()
    # plt.show()
    if save_fig:
        plot_path = f"./output/0218test/plots/{expr}/query_control_{is_query_control}_smaller_range"
        if not os.path.exists(plot_path):
            os.makedirs(plot_path, exist_ok=True)
        plt.savefig(f"{plot_path}/{temp}.pdf")

def save_utopia_and_nadir_w_query_control(methods, queries, stages, cpu_mode, sample_mode, n_c_samples_list, n_p_samples_list,
                          evo_setting, ws_n_samples_list, ppf_setting, save_data_header, is_query_control, is_oracle):
    # methods = ["ws", "evo", "ppf", "div_and_conq_moo%B"]
    for i, (query_id, n_s) in enumerate(zip(queries, stages)):
        all_nadir, all_utopia = get_utopia_nadir_all_algos_w_query_control(methods, query_id, n_s, sample_mode, cpu_mode,
                                                           n_c_samples_list,
                                                           n_p_samples_list,
                                                           evo_setting,
                                                           ws_n_samples_list,
                                                           ppf_setting,
                                                           save_data_header=save_data_header,
                                                           is_query_control=is_query_control,
                                                           is_oracle=is_oracle,
                                                           )
        print(f"the utopia in the query {query_id} is:")
        print(all_utopia)
        print(f"the nadir in the query {query_id} is:")
        print(all_nadir)
        print()
        results = np.vstack((all_utopia, all_nadir))
        save_path = f"{save_data_header}/utopia_and_nadir_w_query_control/query_{query_id}_n_{n_s}"
        save_results(save_path, results=results, mode="utopia_and_nadir")


def compare_approx_seq_div(queries, stages, temp, save_data=False, expr="expr"):
    from matplotlib.ticker import FormatStrFormatter

    fig, ax = plt.subplots()
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.xaxis.set_ticks(np.arange(0.145, 0.165, 0.005))
    # options = ["2_2g_8", "1_1g_16", "5_20g_16"]
    if len(queries) == 1 and "10" in queries[0]:
        # options = ["1_1g_4_po", "1_1g_16_po", "2_2g_16_po"]
        options = ["1_1g_4", "3_3g_16", "5_20g_16"]
        # ax.set_xlim([0.16, 0.26])
        # ax.set_ylim([180, 340])
        # options = ["1_1g_16_po", "1_1g_16_po1", "1_1g_16_po_random2"]
        colors = ["blue", "orange", "green"]
        # colors = ["orange", "cyan", "olive"]
    elif len(queries) == 1 and "2" in queries[0]:
        # options = ["1_1g_4_po", "2_2g_4_po", "1_1g_16_po"]
        # options = ["2_2g_4_po", "2_2g_4_po1", "2_2g_4_po_random2"]
        options = ["1_1g_4", "1_4g_16", "3_3g_16", "5_20g_16"]
        # ax.set_xlim([0.055, 0.085])
        # ax.set_ylim([105, 195])
        colors = ["blue", "orange", "green", "olive", "cyan"]
        # colors = ["orange", "cyan", "olive"]

    for color, option in zip(colors, options):
        approx_F_list = []
        seq_F_list = []
        ws_F_list = []
        for query_id, n_stages in zip(queries, stages):
            data_path1 = f"./output/0218test/compare_diff_res_div_moo/query_control_False/latest_model_cpu/ag/oracle_False/div_and_conq_moo%B/1_256/time_-1/query_2-1_n_14/grid/B/c_{option}"
            data_path2 = f"./output/0218test/compare_diff_res_div_moo/query_control_False/latest_model_cpu/ag/oracle_False/div_and_conq_moo%WS&11/1_256/time_-1/query_2-1_n_14/grid/WS&11/c_{option}"
            data_path3 = f"./output/0218test/compare_diff_res_div_moo/query_control_False/latest_model_cpu/ag/oracle_False/div_and_conq_moo%GD/1_256/time_-1/query_2-1_n_14/grid/GD/c_{option}"
            F_approx = np.loadtxt(f"{data_path1}/F.txt")
            F_ws = np.loadtxt(f"{data_path2}/F.txt")
            F_seq = np.loadtxt(f"{data_path3}/F.txt")
            approx_F_list.append(F_approx)
            ws_F_list.append(F_ws)
            seq_F_list.append(F_seq)
        # all_F_list.append(F_list)

        lat_seq = np.vstack(seq_F_list)[:, 0]
        cost_seq = np.vstack(seq_F_list)[:, 1]
        ax.scatter(cost_seq, lat_seq, marker="s", facecolors='none', edgecolors=color, label=f"GD({option})")

        lat_ws = np.vstack(ws_F_list)[:, 0]
        cost_ws = np.vstack(ws_F_list)[:, 1]
        ax.scatter(cost_ws, lat_ws, marker="o", facecolors='none', edgecolors=color, label=f"WS({option})")

        lat_approx = np.vstack(approx_F_list)[:, 0]
        cost_approx = np.vstack(approx_F_list)[:, 1]
        ax.scatter(cost_approx, lat_approx, marker="*", color="red", label=f"B({option})")

    # data_path_seq_all = f"./output/updated_cpu/div_and_conq_moo/time_-1/query_{queries[0]}_n_{stages[0]}/{sample_mode}/seq_div_and_conq"
    # F_seq_all = np.loadtxt(f"{data_path_seq_all}/F.txt")
    # ax.scatter(F_seq_all[:, 1], F_seq_all[:, 0], marker=".", color="brown", label="GD(All)")
    #
    # data_path_approx = f"./output/updated_cpu/div_and_conq_moo/time_-1/query_{queries[0]}_n_{stages[0]}/{sample_mode}/approx_solve"
    # F_approx = np.loadtxt(f"{data_path_approx}/F.txt")
    # ax.scatter(F_approx[:, 1], F_approx[:, 0], marker=">", color="red")


    ax.set_xlabel("Query Cost", fontdict={"size": 20})
    ax.set_ylabel("Query Latency", fontdict={"size": 20})
    # if len(queries) == 1:
    #     ax.set_title(f"{temp} query {queries[0]} with {stages[0]} subQs")
    # else:
    #     ax.set_title(f"All {len(queries)} {temp} queries")
    plt.xlim([0.145, 0.165])
    plt.ylim([3, 45])
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    ax.legend()
    # plt.show()
    plt.tight_layout()
    if save_data:
        if temp == "tpch":
            plot_path = f"./output/0218test/plots/{expr}/diff_res"
        else:
            plot_path = f"./output0218test/tpcds_traces/plots/{expr}/diff_res"
        if not os.path.exists(plot_path):
            os.makedirs(plot_path, exist_ok=True)
        plt.savefig(f"{plot_path}/diff_res_query_2-1_n_subQ_14.pdf")


    fig1, ax1 = plt.subplots()
    ax1.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax1.xaxis.set_ticks(np.arange(0.145, 0.165, 0.005))

    q_pareto = np.loadtxt(
        "./output/0218test/query_control_False/latest_model_cpu/ag/oracle_False/div_and_conq_moo%B/128_256/time_-1/query_2-1_n_14/grid/B/F.txt")
    lat_q_pareto, cost_q_pareto = q_pareto[:, 0], q_pareto[:, 1]
    ax1.plot(cost_q_pareto, lat_q_pareto, marker="o", color="red", label="Pareto")

    ax1.set_xlabel("Query Cost", fontdict={"size": 20})
    ax1.set_ylabel("Query Latency", fontdict={"size": 20})

    # if len(queries) == 1:
    #     ax.set_title(f"{temp} query {queries[0]} with {stages[0]} subQs")
    # else:
    #     ax.set_title(f"All {len(queries)} {temp} queries")
    plt.xlim([0.145, 0.165])
    plt.ylim([3, 45])
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    ax1.legend()
    # plt.show()
    plt.tight_layout()
    if save_data:
        if temp == "tpch":
            plot_path = f"./output/0218test/plots/{expr}/diff_res"
        else:
            plot_path = f"./output0218test/tpcds_traces/plots/{expr}/diff_res"
        if not os.path.exists(plot_path):
            os.makedirs(plot_path, exist_ok=True)
        plt.savefig(f"{plot_path}/ori_po_query_2-1_n_subQ_14.pdf")


def a_test_moo():
    # plot
    n_samples = 1000
    seed = 0
    np.random.seed(seed)
    latency = np.random.random((n_samples, 1)) * 5
    cost = np.random.random((n_samples, 1)) * 5
    query = np.hstack((latency, cost))
    utopia_lat = min(latency)
    utopia_cost = min(cost)

    fig, ax = plt.subplots()
    ax.set_xlim([-0.2, 5.2])
    ax.set_ylim([-0.2, 5.2])

    ax.scatter(cost, latency, marker=".", color="blue", edgecolors="blue", alpha=0.3, label="Dominated")

    q_pareto_flag = is_pareto_efficient(query)
    q_pareto = query[q_pareto_flag]
    lat_q_pareto, cost_q_pareto = q_pareto[:, 0], q_pareto[:, 1]
    ax.scatter(cost_q_pareto, lat_q_pareto, marker=".", color="red", edgecolors="red", alpha=0.6, label="Pareto")

    ax.scatter([utopia_cost], [utopia_lat], color="orange", edgecolors="orange", alpha=0.6, label="Utopia")
    ax.set_ylabel('Query Latency', fontdict={"size": 20})
    ax.set_xlabel('Query Cost', fontdict={"size": 20})

    weights = np.array([0.7, 0.3])
    rec_f, _ = weighted_utopia_nearest(q_pareto, q_pareto, weights=weights)
    ax.scatter(rec_f[1], rec_f[0], marker="*", color="red", label="Recommendation", s=80)

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    ax.legend(fontsize=20)
    plt.tight_layout()
    # plt.show()

    plot_path = f"./output/plots/example_moo"
    if not os.path.exists(plot_path):
        os.makedirs(plot_path, exist_ok=True)
    plt.savefig(f"{plot_path}/solutions.pdf")


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

def a_test_ws(ws_pairs_list):
    # plot
    n_samples = 1000
    seed = 0
    np.random.seed(seed)
    latency = np.random.random((n_samples, 1)) * 5
    cost = np.random.random((n_samples, 1)) * 5
    query_objs = np.hstack((latency, cost))

    po_obj_list, po_var_list = [], []
    # normalization
    objs_min, objs_max = query_objs.min(0), query_objs.max(0)
    fig, ax = plt.subplots()
    ax.set_xlim([-0.2, 5.2])
    ax.set_ylim([-0.2, 5.2])
    assert all((objs_min - objs_max) <= 0)
    objs_norm = (query_objs - objs_min) / (objs_max - objs_min)
    markers = ["s", "o", "*"]
    marker_size = [80, 60, 40]
    facecolors = ["none", "none", "red"]
    for i, ws_pairs in enumerate(ws_pairs_list):
        for ws in ws_pairs:
            # po_ind = self.get_soo_index(objs_norm, ws)
            obj = np.sum(objs_norm * ws, axis=1)
            po_ind = np.argmin(obj)
            po_obj_list.append(query_objs[po_ind])

        # only keep non-dominated solutions
        po_objs_arr = np.array(po_obj_list)
        q_pareto_flag = is_pareto_efficient(po_objs_arr)
        q_pareto = po_objs_arr[q_pareto_flag]

        print()


        lat_q_pareto, cost_q_pareto = q_pareto[:, 0], q_pareto[:, 1]
        ax.scatter(cost_q_pareto, lat_q_pareto, marker=markers[i], facecolors=facecolors[i], color="red", edgecolors="red", s=marker_size[i], label=f"Pareto (n_ws={len(ws_pairs)})")

        ax.set_ylabel('Query Latency', fontdict={"size": 20})
        ax.set_xlabel('Query Cost', fontdict={"size": 20})

        # weights = np.array([0.7, 0.3])
        # rec_f, _ = weighted_utopia_nearest(q_pareto, q_pareto, weights=weights)
        # ax.scatter(rec_f[1], rec_f[0], marker="*", color="red", label="Recommendation", s=80)

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    ax.legend(fontsize=20)
    plt.tight_layout()
    # plt.show()
    plot_path = f"./output/plots/example_moo"
    if not os.path.exists(plot_path):
        os.makedirs(plot_path, exist_ok=True)
    plt.savefig(f"{plot_path}/ws.pdf")


def plot_ws_pareto_frontier(queries, stages, n_samples, save_data_header,
                            is_query_control, cpu_mode, model_name, is_oracle):

    n_ws_list = [11, 21, 51, 101]

    colors_auto = cm.rainbow(np.linspace(0, 1, len(n_ws_list)))
    for query_id, n_stages in zip(queries, stages):
        fig, ax = plt.subplots()
        for i, n_ws in enumerate(n_ws_list):
            data_path = f"{save_data_header}/query_control_{is_query_control}/" \
                        f"latest_model_{cpu_mode}/{model_name}/oracle_{is_oracle}/" \
                        f"ws/{n_samples}_{n_ws}/time_-1/query_{query_id}_n_{n_stages}"

            F = np.loadtxt(f"{data_path}/F.txt")
            ax.scatter(F[:, 1], F[:, 0], color=colors_auto[i], label=f"n_ws = {n_ws}")
            ax.legend()

        ax.set_xlabel(f"Query Cost")
        ax.set_ylabel(f"Query Latency")
        ax.set_title(f"Query {query_id} with {n_stages} subQs")
        plt.tight_layout()
        plt.show()

        print()

def plot_example_query_ws():
    data_header = f"./output/0218test/n_ws_test/query_control_False/latest_model_cpu/ag/oracle_False/ws"
    n_ws_list = [11, 21, 51, 101]

    colors_auto = cm.rainbow(np.linspace(0, 1, len(n_ws_list)))

    for i, n_ws in enumerate(n_ws_list):
        data_path = f"{data_header}/100000_{n_ws}/time_-1/query_2-1_n_14"
        F = np.loadtxt(f"{data_path}/F.txt")

        fig, ax = plt.subplots()
        ax.scatter(F[:, 1], F[:, 0], color="blue", edgecolors="blue", alpha=0.3, label=f"n_ws={n_ws}")

        rec_1, _ = weighted_utopia_nearest(F, F, weights=np.array([0.1, 0.9]))
        ax.scatter([rec_1[1]], [rec_1[0]], marker="s", color="orange", s=80, label=f"rec_[0.1, 0.9]")

        rec_2, _ = weighted_utopia_nearest(F, F, weights=np.array([0.9, 0.1]))
        ax.scatter([rec_2[1]], [rec_2[0]], marker="o", color="olive", s=60, label=f"rec_[0.9, 0.1]")

        rec_3, _ = weighted_utopia_nearest(F, F, weights=np.array([0.5, 0.5]))
        ax.scatter([rec_3[1]], [rec_3[0]], marker="+", color="red", s=40, label=f"rec_[0.5, 0.5]")


        ax.legend()

        ax.set_xlabel(f"Query Cost")
        ax.set_ylabel(f"Query Latency")
        ax.set_title(f"TPCH Query 2-1 with 14 subQs")
        plt.tight_layout()
        # plt.show()
        plot_path = f"./output/0218test/plots/example_ws_tpch"
        if not os.path.exists(plot_path):
            os.makedirs(plot_path, exist_ok=True)
        plt.savefig(f"{plot_path}/n_ws_{n_ws}.pdf")


def plot_example_query_ws_vs_ours(save_data_header):

    data_path_ws = f"{save_data_header}/ws/100000_11/time_-1/query_2-1_n_14"
    F_ws = np.loadtxt(f"{data_path_ws}/F.txt")

    data_path_div_moo = f"{save_data_header}/div_and_conq_moo%B/128_256/time_-1/query_2-1_n_14/grid/B"
    F_div_moo = np.loadtxt(f"{data_path_div_moo}/F.txt")

    fig, ax = plt.subplots()
    ax.scatter(F_ws[:, 1], F_ws[:, 0], color="purple", edgecolors="purple", alpha=0.5, label=f"ws")

    ws_rec_1, _ = weighted_utopia_nearest(F_ws, F_ws, weights=np.array([0.1, 0.9]))
    ax.scatter([ws_rec_1[1]], [ws_rec_1[0]], color="blue", marker="s", s=80, label=f"ws_[0.1, 0.9]")

    ws_rec_2, _ = weighted_utopia_nearest(F_ws, F_ws, weights=np.array([0.9, 0.1]))
    ax.scatter([ws_rec_2[1]], [ws_rec_2[0]], color="blue", marker=">", s=60, label=f"ws_[0.9, 0.1]")

    ws_rec_3, _ = weighted_utopia_nearest(F_ws, F_ws, weights=np.array([0.5, 0.5]))
    ax.scatter([ws_rec_3[1]], [ws_rec_3[0]], color="blue", marker="*", s=150, label=f"ws_[0.5, 0.5]")

    ax.scatter(F_div_moo[:, 1], F_div_moo[:, 0], color="pink", edgecolors="pink", alpha=0.5, label=f"ours")

    div_rec_1, _ = weighted_utopia_nearest(F_div_moo, F_div_moo, weights=np.array([0.1, 0.9]))
    ax.scatter([div_rec_1[1]], [div_rec_1[0]], color="red", marker="s", s=80, label=f"ours_[0.1, 0.9]")

    div_rec_2, _ = weighted_utopia_nearest(F_div_moo, F_div_moo, weights=np.array([0.9, 0.1]))
    ax.scatter([div_rec_2[1]], [div_rec_2[0]], color="red", marker=">", s=60, label=f"ours_[0.9, 0.1]")

    div_rec_3, _ = weighted_utopia_nearest(F_div_moo, F_div_moo, weights=np.array([0.5, 0.5]))
    ax.scatter([div_rec_3[1]], [div_rec_3[0]], color="red", marker="*", s=120, label=f"ours_[0.5, 0.5]")

    ax.legend()

    ax.set_xlabel(f"Query Cost")
    ax.set_ylabel(f"Query Latency")
    ax.set_title(f"TPCH Query 2-1 with 14 subQs")
    plt.tight_layout()
    # plt.show()
    plot_path = f"./output/0218test/plots/example_ours_ws_tpch"
    if not os.path.exists(plot_path):
        os.makedirs(plot_path, exist_ok=True)
    plt.savefig(f"{plot_path}/n_ws_11.pdf")

def plot_compile_time_opt_example():
    lat_q = [45.1, 32.6, 30.6]
    cost_q  = [21.8, 23.4, 36.4]
    fig, ax = plt.subplots()
    ax.plot(cost_q, lat_q, marker="o", color="blue")

    dominated_q_lat = [40.1]
    dominated_q_cost = [24.8]
    ax.scatter(dominated_q_cost, dominated_q_lat, marker="o", color="red")

    ax.set_ylabel('Latency', fontdict={"size": 20})
    ax.set_xlabel('Cost', fontdict={"size": 20})
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    # ax.legend(fontsize=20)
    plt.tight_layout()
    plt.show()


def plot_frontier_one_query():
    data_path = f"./output/0218test/tpch_traces/new_test_end/query_control_False/latest_model_cpu/ag/oracle_False/div_and_conq_moo%B/256_256/time_-1/query_5-1_n_11/grid/B"
    F = np.loadtxt(f"{data_path}/F.txt")

    fix, ax = plt.subplots()
    ax.scatter(F[:, 1], F[:, 0])
    ax.set_xlabel("Query Cost")
    ax.set_ylabel("Query Latency")
    plt.tight_layout()
    plt.show()

def expr5(temp, queries, stages, save_fig=False):


    # div_moo_setting_tpch = "128_256"  # tpch
    # save_data_header_tpch = f"./output/0218test"
    # div_moo_setting_tpcds = "128_32"  # tpcds
    # save_data_header_tpcds = f"./output/0218test/tpcds_traces"

    if temp == "tpch":
        div_moo_setting = "128_256"  # tpch
        save_data_header = f"./output/0218test"
    else:
        div_moo_setting = "128_32"  # tpcds
        save_data_header = f"./output/0218test/tpcds_traces"

    cpu_mode = "cpu"
    sample_mode = "grid"
    is_oracle = False
    model_name = "ag"
    expr = "expr5"

    # compare_diff_dag_opt(queries, stages, cpu_mode=cpu_mode,
    #                      sample_mode=sample_mode,
    #                      div_moo_setting=div_moo_setting_tpcds,
    #                      save_data_header=save_data_header_tpcds,
    #                      is_oracle=is_oracle,
    #                      is_query_control=False,
    #                      save_pareto_front=True,
    #                      model_name=model_name,
    #                      save_fig=save_fig,
    #                      expr=expr,
    #                      temp=temp)

    compare_approx_seq_div([queries_tpch[1]], [stages_tpch[1]], temp="tpch", save_data=True, expr=expr)

def expr6(temp,
          queries,
          stages,
          save_fig):
    # div_moo_setting_tpch = "128_256"  # tpch
    # save_data_header_tpch = f"./output/0218test"
    # div_moo_setting_tpcds = "128_32"  # tpcds
    # save_data_header_tpcds = f"./output/0218test/tpcds_traces"
    if temp == "tpch":
        div_moo_setting = "128_256"  # tpch
        save_data_header = f"./output/0218test"
        # div_moo_setting = "160_162"  # tpch
        # save_data_header = f"./output/0218test/tpch_traces/run_new_grid_new_s"
    else:
        div_moo_setting = "128_32"  # tpcds
        save_data_header = f"./output/0218test/tpcds_traces"
        # div_moo_setting = "36_24"  # tpcds
        # save_data_header = f"./output/0218test/tpcds_traces/run_new_grid_new_s"
    cpu_mode = "cpu"
    sample_mode = "grid"
    is_oracle = False
    model_name = "ag"
    expr = "expr6"

    compare_query_finer_control_HV(queries, stages, cpu_mode, sample_mode,
                                   save_data_header=save_data_header,
                                   evo_setting=evo_setting,
                                   ws_setting=ws_setting,
                                   ppf_setting=ppf_setting,
                                   div_moo_setting=div_moo_setting,
                                   is_query_control=False,
                                   is_oracle=is_oracle,
                                   model_name=model_name,
                                   temp=temp,
                                   save_fig=save_fig,
                                   expr=expr)

def expr7(temp,
          queries,
          stages,
          save_fig):
    if temp == "tpch":
        div_moo_setting = "128_256"  # tpch
        save_data_header = f"./output/0218test"
    else:
        div_moo_setting = "128_32"  # tpcds
        save_data_header = f"./output/0218test/tpcds_traces"

    cpu_mode = "cpu"
    sample_mode = "grid"
    is_oracle = False
    model_name = "ag"
    expr = "expr7"
    compare_query_finer_control_HV(queries, stages, cpu_mode, sample_mode,
                                   save_data_header=save_data_header,
                                   evo_setting=evo_setting,
                                   ws_setting=ws_setting,
                                   ppf_setting=ppf_setting,
                                   div_moo_setting=div_moo_setting,
                                   is_query_control=True,
                                   is_oracle=is_oracle,
                                   model_name=model_name,
                                   temp=temp,
                                   save_fig=save_fig,
                                   expr=expr)


    if temp == "tpch":
        ws_n_samples_list = [1000, 10_000, 100_000, 1_000_000, 10_000_000]
        # methods = ["ws", "evo", "ppf", "div_and_conq_moo%B"]
        methods = ["ws"]
        save_data_header_smaller = f"./output/0218test/smaller_range"
        n_stages_upper = 15
        n_stages_lower = 1
        inds = [i for i, k in enumerate(stages) if k >= n_stages_lower and k <= n_stages_upper]

        queries_test = [queries[i] for i in inds]
        stages_test = [stages[i] for i in inds]

        compare_fine_query_control_diff_setting(queries_test,
                                                stages_test,
                                                save_data_header=save_data_header_smaller,
                                                cpu_mode=cpu_mode,
                                                is_oracle=False,
                                                ws_n_samples_list=ws_n_samples_list,
                                                save_fig=save_fig,
                                                temp=temp,
                                                expr=expr)

def get_all_hv_tc_expr5(methods,
                        queries,
                        stages,
                        cpu_mode,
                        sample_mode,
                        div_moo_setting,
                        save_data_header="",
                        save_pareto_front=False,
                         model_name="mlp",
                        expr="expr",
                        ):
    # methods = ["div_and_conq_moo%GD", "div_and_conq_moo%WS&11", "div_and_conq_moo%B"]

    all_tc_list = []
    all_hv_list = []

    for query_id, n_s in zip(queries, stages):
        F_list = []
        tc_list = []
        algo_labels = []
        utopia_and_nadir_path = f"{save_data_header}/utopia_and_nadir/query_{query_id}_n_{n_s}"
        utopia_and_nadir = np.loadtxt(f"{utopia_and_nadir_path}/utopia_and_nadir.txt")
        all_utopia = utopia_and_nadir[0, :]
        all_nadir = utopia_and_nadir[1, :]

        if save_pareto_front:
            # fig, ax = plt.subplots()
            plt.figure(figsize=(3.5, 2.5))
            colors = ["blue", "orange", "green"]
            markers = ["s", "o", "*"]

        for i, method in enumerate(methods):
            dag_opt_algo = method.split("%")[1]
            # div_moo_setting = f"{n_c_samples}_{n_p_samples}"

            if "WS" in dag_opt_algo:
                # data_path = f"./output/0218test/norm_div_WS/query_control_False/latest_model_{cpu_mode}/ag/oracle_False/{method}/" \
                #             f"{div_moo_setting}/time_-1/query_{query_id}_n_{n_s}/{sample_mode}/{dag_opt_algo}"
                data_path = f"{save_data_header}/query_control_False/latest_model_{cpu_mode}/{model_name}/oracle_False/{method}/" \
                            f"{div_moo_setting}/time_-1/query_{query_id}_n_{n_s}/{sample_mode}/{dag_opt_algo}"
                algo_labels.append("WS-based")
            else:
                data_path = f"{save_data_header}/query_control_False/latest_model_{cpu_mode}/{model_name}/oracle_False/{method}/" \
                            f"{div_moo_setting}/time_-1/query_{query_id}_n_{n_s}/{sample_mode}/{dag_opt_algo}"
                algo_labels.append(dag_opt_algo)

            F = np.loadtxt(f"{data_path}/F.txt")
            F_uniq = np.around(np.unique(F.reshape(-1, 2), axis=0), 5)
            F_list.append(F_uniq)
            # tc = np.loadtxt(f"{data_path}/time.txt")
            # tc_list.append(tc.item())
            tc_dict_file = f"{data_path}/end_to_end.json"
            with open(tc_dict_file) as f:
                d = json.load(f)
            tc = d["end_to_end"]
            tc_list.append(tc)

            if save_pareto_front:
                plt.scatter(F[:, 1], F[:, 0], marker=markers[i], color=colors[i],
                           label=f"{algo_labels[i]} ({F_uniq.shape[0]})")
                plt.legend()
                plt.xlabel("Query Cost")
                plt.ylabel("Query Analytical Latency")
                # plt.set_title(f"Pareto frontier of query {query_id} with {n_s} subQs")
                plt.tight_layout()
                # plt.show()
                plot_path = f"{save_data_header}/plots/{expr}/compare_dag_opt"
                if not os.path.exists(plot_path):
                    os.makedirs(plot_path, exist_ok=True)
                plt.savefig(f"{plot_path}/query_{query_id}_n_subQ_{n_s}.pdf")
        hv = cal_hv(F_list, all_nadir, all_utopia)

        all_tc_list.append(tc_list)

        if max(hv) > 100:
            print("There is something wrong with the reference points!")

        all_hv_list.append(hv)

    hv_mean = np.mean(np.array(all_hv_list), axis=0)
    tc_mean = np.mean(np.array(all_tc_list), axis=0)

    hv_std = np.std(np.array(all_hv_list), axis=0)
    tc_std = np.std(np.array(all_tc_list), axis=0)

    hv_min = np.min(np.array(all_hv_list), axis=0)
    tc_min = np.min(np.array(all_tc_list), axis=0)

    hv_max = np.max(np.array(all_hv_list), axis=0)
    tc_max = np.max(np.array(all_tc_list), axis=0)

    print(f"mean HV with mode {cpu_mode} is {hv_mean}")
    print(f"std HV with mode {cpu_mode} is {hv_std}")
    print(f"min HV with mode {cpu_mode} is {hv_min}")
    print(f"max HV with mode {cpu_mode} is {hv_max}")
    print(f"mean solving time with mode {cpu_mode} is {tc_mean}")
    print(f"std solving time with mode {cpu_mode} is {tc_std}")
    print(f"min solving time with mode {cpu_mode} is {tc_min}")
    print(f"max solving time with mode {cpu_mode} is {tc_max}")
    print()

    return hv_mean, hv_std, all_tc_list

def get_all_hv_tc_expr6_7(methods,
                        queries,
                        stages,
                        cpu_mode,
                        sample_mode,
                        div_moo_setting,
                        save_data_header="",
                        save_pareto_front=False,
                        model_name="mlp",
                        expr="expr",
                        is_query_control=False,
                          ):
    # methods = ["ws", "evo", "ppf", "div_and_conq_moo%B"]
    algo_labels = []

    all_F_list = []
    all_hv_list, all_tc_list = [], []
    for i, (query_id, n_s) in enumerate(zip(queries, stages)):
        utopia_and_nadir_path = f"{save_data_header}/utopia_and_nadir/query_{query_id}_n_{n_s}"
        utopia_and_nadir = np.loadtxt(f"{utopia_and_nadir_path}/utopia_and_nadir.txt")
        all_utopia = utopia_and_nadir[0, :]
        all_nadir = utopia_and_nadir[1, :]

        F_list = []
        hv_list, tc_list = [], []
        for method in methods:
            if method == "evo":
                data_path = f"{save_data_header}/query_control_{is_query_control}/latest_model_{cpu_mode}/{model_name}/oracle_{is_oracle}/{method}/{evo_setting}/time_-1/query_{query_id}_n_{n_s}"
                F = np.loadtxt(f"{data_path}/F.txt")

                F_list.append(np.around(np.unique(F.reshape(-1, 2), axis=0), 5))

                tc_dict_file = f"{data_path}/end_to_end.json"
                with open(tc_dict_file) as f:
                    d = json.load(f)
                tc = d["end_to_end"]
                tc_list.append(tc)
                if i == 0:
                    algo_labels.append(method)
            elif method == "ws":
                data_path = f"{save_data_header}/query_control_{is_query_control}/latest_model_{cpu_mode}/{model_name}/oracle_{is_oracle}/{method}/{ws_setting}/time_-1/query_{query_id}_n_{n_s}"
                F = np.loadtxt(f"{data_path}/F.txt")

                F_list.append(np.around(np.unique(F.reshape(-1, 2), axis=0), 5))

                tc_dict_file = f"{data_path}/end_to_end.json"
                with open(tc_dict_file) as f:
                    d = json.load(f)
                tc = d["end_to_end"]
                tc_list.append(tc)
                if i == 0:
                    algo_labels.append(method)
            elif method == "ppf":
                # if is_query_control:
                #     data_path = f"./output/test/query_control_True/latest_model_{cpu_mode}/ag/ppf/{ppf_setting}/time_-1/query_{query_id}_n_{n_s}"
                # else:
                #     data_path = f"./output/test/query_control_False/latest_model_{cpu_mode}/ag/ppf/{ppf_setting}/time_-1/query_{query_id}_n_{n_s}"
                data_path = f"{save_data_header}/query_control_{is_query_control}/latest_model_{cpu_mode}/{model_name}/oracle_{is_oracle}/{method}/{ppf_setting}/time_-1/query_{query_id}_n_{n_s}"

                F = np.loadtxt(f"{data_path}/F.txt")

                F_list.append(np.around(np.unique(F.reshape(-1, 2), axis=0), 5))
                # tc = np.loadtxt(f"{data_path}/time.txt")
                # tc_list.append(tc.item())
                tc_dict_file = f"{data_path}/end_to_end.json"
                with open(tc_dict_file) as f:
                    d = json.load(f)
                tc = d["end_to_end"]
                tc_list.append(tc)
                if i == 0:
                    algo_labels.append(method)
            else:

                dag_opt_algo = method.split("%")[1]
                data_path = f"{save_data_header}/query_control_False/latest_model_{cpu_mode}/{model_name}/oracle_False/{method}/" \
                            f"{div_moo_setting}/time_-1/query_{query_id}_n_{n_s}/{sample_mode}/{dag_opt_algo}"
                F = np.loadtxt(f"{data_path}/F.txt")

                F_list.append(np.around(np.unique(F.reshape(-1, 2), axis=0), 5))
                # tc = np.loadtxt(f"{data_path}/time.txt")
                # tc_list.append(tc.item())
                tc_dict_file = f"{data_path}/end_to_end.json"
                with open(tc_dict_file) as f:
                    d = json.load(f)
                tc = d["end_to_end"]
                tc_list.append(tc)
                if i == 0:
                    algo_labels.append(f"ours")

                #
        hv = cal_hv(F_list, all_nadir, all_utopia)

        if max(hv) > 100:
            print("There is something wrong with the reference points!")

        hv_list.extend(hv)

        all_hv_list.append(hv_list)
        all_tc_list.append(tc_list)
        all_F_list.append(F_list)

    hv_mean = np.mean(np.array(all_hv_list), axis=0)
    tc_mean = np.mean(np.array(all_tc_list).reshape(-1, hv_mean.shape[0]), axis=0)

    hv_std = np.std(np.array(all_hv_list), axis=0)
    tc_std = np.std(np.array(all_tc_list), axis=0)

    hv_min = np.min(np.array(all_hv_list), axis=0)
    tc_min = np.min(np.array(all_tc_list), axis=0)

    hv_max = np.max(np.array(all_hv_list), axis=0)
    tc_max = np.max(np.array(all_tc_list), axis=0)

    print(f"mean HV with mode {cpu_mode} is {hv_mean}")
    print(f"std HV with mode {cpu_mode} is {hv_std}")
    print(f"min HV with mode {cpu_mode} is {hv_min}")
    print(f"max HV with mode {cpu_mode} is {hv_max}")
    print(f"mean solving time with mode {cpu_mode} is {tc_mean}")
    print(f"std solving time with mode {cpu_mode} is {tc_std}")
    print(f"min solving time with mode {cpu_mode} is {tc_min}")
    print(f"max solving time with mode {cpu_mode} is {tc_max}")
    print()
    our_vs_ws_hv = (hv_mean[-1] - hv_mean[0]) / hv_mean[0]
    our_vs_evo_hv = (hv_mean[-1] - hv_mean[1]) / hv_mean[1]
    our_vs_ppf_hv = (hv_mean[-1] - hv_mean[2]) / hv_mean[2]
    print(f"the improvement of HV v.s. (ws, evo, ppf) is: "
          f"{our_vs_ws_hv}, {our_vs_evo_hv}, {our_vs_ppf_hv}")

    our_vs_ws_tc = (tc_mean[0] - tc_mean[-1]) / tc_mean[0]
    our_vs_evo_tc = (tc_mean[1] - tc_mean[-1]) / tc_mean[1]
    our_vs_ppf_tc = (tc_mean[2] - tc_mean[-1]) / tc_mean[2]
    print(f"the improvement of Time cost v.s. (ws, evo, ppf) is: "
          f"{our_vs_ws_tc}, {our_vs_evo_tc}, {our_vs_ppf_tc}")

    return hv_mean, hv_std, all_tc_list


def expr5_plus(save_fig):
    # for TPCH
    queries_tpch = [f"{i}-1" for i in np.arange(1, 23)]
    stages_tpch = [3, 14, 5, 5, 11, 2, 11, 13, 12, 7, 10, 5, 5, 4, 7, 7, 7, 9, 5, 14, 12, 7]

    # for TPCDS
    save_data_header_tpcds = f"./output/0218test/tpcds_traces"
    query_id_path = f"{save_data_header_tpcds}/query_info/query_id_list.txt"
    n_stages_path = f"{save_data_header_tpcds}/query_info/n_stages_list.txt"

    queries_tpcds = np.loadtxt(f"{query_id_path}", delimiter=" ", dtype=str).tolist()
    stages_tpcds = np.loadtxt(f"{n_stages_path}").astype(int).tolist()

    div_moo_setting_tpch = "128_256"  # tpch
    save_data_header_tpch = f"./output/0218test"
    div_moo_setting_tpcds = "128_32"  # tpcds
    save_data_header_tpcds = f"./output/0218test/tpcds_traces"
    cpu_mode = "cpu"
    sample_mode = "grid"
    model_name = "ag"
    expr = "expr5"
    methods = ["div_and_conq_moo%GD", "div_and_conq_moo%WS&11", "div_and_conq_moo%B"]
    temp_list = ["TPCH", "TPCDS"]
    algo_labels = ["GD", "WSA", "BA"]

    hv_mean_tpch, hv_std_tpch, all_tc_list_tpch = get_all_hv_tc_expr5(
                                                            methods=methods,
                                                            queries=queries_tpch,
                                                            stages=stages_tpch,
                                                            cpu_mode=cpu_mode,
                                                            sample_mode=sample_mode,
                                                            div_moo_setting=div_moo_setting_tpch,
                                                            save_data_header=save_data_header_tpch,
                                                            save_pareto_front=False,
                                                            model_name=model_name,
                                                            expr=expr)

    hv_mean_tpcds, hv_std_tpcds, all_tc_list_tpcds = get_all_hv_tc_expr5(
        methods=methods,
        queries=queries_tpcds,
        stages=stages_tpcds,
        cpu_mode=cpu_mode,
        sample_mode=sample_mode,
        div_moo_setting=div_moo_setting_tpcds,
        save_data_header=save_data_header_tpcds,
        save_pareto_front=False,
        model_name=model_name,
        expr=expr)



    x = np.repeat(temp_list, len(algo_labels)).tolist()

    hv_mean_all = np.round(hv_mean_tpch, 2).tolist() + np.round(hv_mean_tpcds, 3).tolist()
    hv_std_all = np.round(hv_std_tpch, 2).tolist() + np.round(hv_std_tpcds, 3).tolist()

    category = np.tile(algo_labels, (len(temp_list), 1)).flatten().tolist()

    raw_data = {'Queries': x,
                "HyperVolume (%)": hv_mean_all,
                "hv_std": hv_std_all,
                "DAG Aggregation": category}
    sns.set(rc={'figure.figsize': (6.5, 4.5)}) # hi
    sns.set(font_scale=1.3)
    sns.set_theme(style="ticks")
    ax = sns.barplot(x="Queries",
                     y="HyperVolume (%)",
                     hue="DAG Aggregation",
                     data=raw_data,
                     errorbar="sd")
    ax.set(xlabel=None)
    for container in ax.containers:
        ax.bar_label(container)
    sns.move_legend(ax, "lower right")
    plt.tight_layout()
    plt.show()

    figure = ax.get_figure()
    if save_fig:
        plot_path = f"./output/final_expts/plots/{expr}/query_control_False"
        if not os.path.exists(plot_path):
            os.makedirs(plot_path, exist_ok=True)
        figure.savefig(f"{plot_path}/HV.pdf")

    # g = sns.catplot(kind='box', data=df_long, x='a', y='c', hue='b', height=5, aspect=1)

    columns = ["Time (s)", "Queries", "DAG Aggregation"]
    algo_tpch = np.tile(algo_labels, (len(all_tc_list_tpch), 1)).flatten()
    temp_tpch = np.array([temp_list[0]] * len(all_tc_list_tpch) * len(algo_labels))
    tc_tpch_arr = np.vstack((np.array(all_tc_list_tpch).flatten(), temp_tpch, algo_tpch)).T
    tc_tpch_df = pd.DataFrame(data=tc_tpch_arr, columns=columns, index=np.arange(tc_tpch_arr.shape[0]))

    algo_tpcds = np.tile(algo_labels, (len(all_tc_list_tpcds), 1)).flatten()
    temp_tpcds = np.array([temp_list[1]] * len(all_tc_list_tpcds) * len(algo_labels))
    tc_tpcds_arr = np.vstack((np.array(all_tc_list_tpcds).flatten(), temp_tpcds, algo_tpcds)).T
    tc_tpcds_df = pd.DataFrame(data=tc_tpcds_arr, columns=columns, index=np.arange(tc_tpch_arr.shape[0],
                                                                                   tc_tpch_arr.shape[0] + tc_tpcds_arr.shape[0]))

    tc_all_df = pd.concat([tc_tpch_df, tc_tpcds_df], axis=0)

    tc_all_df["Time (s)"] = tc_all_df["Time (s)"].astype(np.float64)
    ax1 = sns.boxplot(x="Queries",
                y="Time (s)",
                hue="DAG Aggregation",
                data=tc_all_df,
                log_scale=True)
    ax1.set(xlabel=None)
    plt.tight_layout()
    plt.show()
    figure1 = ax1.get_figure()
    if save_fig:
        plot_path = f"./output/final_expts/plots/{expr}/query_control_False"
        if not os.path.exists(plot_path):
            os.makedirs(plot_path, exist_ok=True)
        figure1.savefig(f"{plot_path}/Time.pdf")

def expr6_7_plus(temp="TPCH", save_fig=False):

    if temp == "TPCH":
        # for TPCH
        queries = [f"{i}-1" for i in np.arange(1, 23)]
        stages = [3, 14, 5, 5, 11, 2, 11, 13, 12, 7, 10, 5, 5, 4, 7, 7, 7, 9, 5, 14, 12, 7]
        div_moo_setting = "128_256"  # tpch
        save_data_header = f"./output/0218test"
    else:

        # for TPCDS
        save_data_header = f"./output/0218test/tpcds_traces"
        query_id_path = f"{save_data_header}/query_info/query_id_list.txt"
        n_stages_path = f"{save_data_header}/query_info/n_stages_list.txt"

        queries = np.loadtxt(f"{query_id_path}", delimiter=" ", dtype=str).tolist()
        stages = np.loadtxt(f"{n_stages_path}").astype(int).tolist()

        div_moo_setting = "128_32"  # tpcds

    cpu_mode = "cpu"
    sample_mode = "grid"
    model_name = "ag"
    expr = "expr6_7"
    methods_finer = ["ws", "evo", "ppf", "div_and_conq_moo%B"]
    methods_query = ["ws", "evo", "ppf"]

    control_list = ["SubQ", "Query"]
    algo_labels_finer = ["WS", "EVO", "PF", "HMOOC3"]
    algo_labels_query = ["WS", "EVO", "PF"]

    hv_mean_fine, hv_std_fine, all_tc_list_fine = get_all_hv_tc_expr6_7(
                                                            methods=methods_finer,
                                                            queries=queries,
                                                            stages=stages,
                                                            cpu_mode=cpu_mode,
                                                            sample_mode=sample_mode,
                                                            div_moo_setting=div_moo_setting,
                                                            save_data_header=save_data_header,
                                                            save_pareto_front=False,
                                                            model_name=model_name,
                                                            expr=expr,
                                                            is_query_control=False)

    hv_mean_query, hv_std_query, all_tc_list_query = get_all_hv_tc_expr6_7(
        methods=methods_query,
        queries=queries,
        stages=stages,
        cpu_mode=cpu_mode,
        sample_mode=sample_mode,
        div_moo_setting=div_moo_setting,
        save_data_header=save_data_header,
        save_pareto_front=False,
        model_name=model_name,
        expr=expr,
        is_query_control=True)

    sns.set_theme(style="ticks")
    # for HV
    columns_hv = ["HyperVolume (%)", "Control", f"MOO algos in {temp}"]
    algo_fine_hv = algo_labels_finer
    control_fine_hv = np.array([control_list[0]] * hv_mean_fine.shape[0])
    hv_fine_arr = np.vstack((np.round(hv_mean_fine, 2), control_fine_hv, algo_fine_hv)).T
    hv_fine_df = pd.DataFrame(data=hv_fine_arr, columns=columns_hv, index=np.arange(hv_fine_arr.shape[0]))

    algo_query_hv = algo_labels_query
    control_query_hv = np.array([control_list[1]] * hv_mean_query.shape[0])
    hv_query_arr = np.vstack((np.round(hv_mean_query, 2), control_query_hv, algo_query_hv)).T
    hv_query_df = pd.DataFrame(data=hv_query_arr, columns=columns_hv, index=np.arange(hv_fine_arr.shape[0],
                                                                                   hv_fine_arr.shape[0] +
                                                                                   hv_query_arr.shape[0]))

    hv_all_df = pd.concat([hv_fine_df, hv_query_df], axis=0)

    hv_all_df["HyperVolume (%)"] = hv_all_df["HyperVolume (%)"].astype(np.float64)
    ax = sns.barplot(x=f"MOO algos in {temp}",
                      y="HyperVolume (%)",
                      hue="Control",
                      data=hv_all_df,
                     errorbar="sd")
    for container in ax.containers:
        ax.bar_label(container)
    sns.move_legend(ax, "lower right")
    plt.tight_layout()
    plt.show()
    figure = ax.get_figure()
    if save_fig:
        plot_path = f"./output/final_expts/plots/{expr}/query_control_False/{temp}"
        if not os.path.exists(plot_path):
            os.makedirs(plot_path, exist_ok=True)
        figure.savefig(f"{plot_path}/HV.pdf")

    # for time cost
    columns = ["Time (s)", "Control", f"MOO algos in {temp}"]
    algo_fine = np.tile(algo_labels_finer, (len(all_tc_list_fine), 1)).flatten()
    control_fine = np.array([control_list[0]] * len(all_tc_list_fine) * len(algo_labels_finer))
    tc_fine_arr = np.vstack((np.array(all_tc_list_fine).flatten(), control_fine, algo_fine)).T
    tc_fine_df = pd.DataFrame(data=tc_fine_arr, columns=columns, index=np.arange(tc_fine_arr.shape[0]))

    algo_query = np.tile(algo_labels_query, (len(all_tc_list_query), 1)).flatten()
    control_query = np.array([control_list[1]] * len(all_tc_list_query) * len(algo_labels_query))
    tc_query_arr = np.vstack((np.array(all_tc_list_query).flatten(), control_query, algo_query)).T
    tc_query_df = pd.DataFrame(data=tc_query_arr, columns=columns, index=np.arange(tc_fine_arr.shape[0],
                                                                                   tc_fine_arr.shape[0] + tc_query_arr.shape[0]))

    tc_all_df = pd.concat([tc_fine_df, tc_query_df], axis=0)

    tc_all_df["Time (s)"] = tc_all_df["Time (s)"].astype(np.float64)
    ax1 = sns.boxplot(x=f"MOO algos in {temp}",
                y="Time (s)",
                hue="Control",
                data=tc_all_df,
                log_scale=True)
    plt.tight_layout()
    plt.show()
    figure1 = ax1.get_figure()
    if save_fig:
        plot_path = f"./output/final_expts/plots/{expr}/query_control_False/{temp}"
        if not os.path.exists(plot_path):
            os.makedirs(plot_path, exist_ok=True)
        figure1.savefig(f"{plot_path}/Time.pdf")

if __name__ == "__main__":
    # for TPCH
    queries_tpch = [f"{i}-1" for i in np.arange(1, 23)]
    stages_tpch = [3, 14, 5, 5, 11, 2, 11, 13, 12, 7, 10, 5, 5, 4, 7, 7, 7, 9, 5, 14, 12, 7]

    # for TPCDS
    save_data_header_tpcds = f"./output/0218test/tpcds_traces"
    query_id_path = f"{save_data_header_tpcds}/query_info/query_id_list.txt"
    n_stages_path = f"{save_data_header_tpcds}/query_info/n_stages_list.txt"

    queries_tpcds = np.loadtxt(f"{query_id_path}", delimiter=" ", dtype=str).tolist()
    stages_tpcds = np.loadtxt(f"{n_stages_path}").astype(int).tolist()

    evo_setting = "100_500"
    ws_setting = "100000_11"
    ppf_setting = "1_2_2"
    is_oracle = False
    is_query_control = True

    # expr5(temp="tpch",
    #       queries=queries_tpch,
    #       stages=stages_tpch,
    #       save_fig=True)
    # expr6(temp="tpcds",
    #       queries=queries_tpcds,
    #       stages=stages_tpcds,
    #       save_fig=True)

    # expr7(temp="tpch",
    #       queries=queries_tpch,
    #       stages=stages_tpch,
    #       save_fig=True)

    # for final plots in paper
    expr5_plus(save_fig=True)

    expr6_7_plus(temp="TPCDS", save_fig=True)




