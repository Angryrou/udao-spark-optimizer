# Copyright (c) 2020 École Polytechnique
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Author(s): Qi FAN <qi dot fan at polytechnique dot edu>
#
# Description: An example of missing solutions
#
# Created at 14/06/2023
import numpy as np
import matplotlib.pyplot as plt
from udao_spark.optimizer.utils import is_pareto_efficient

import os

def plot_stage(i, stage, m, c, save_fig=False):
    # fig, ax = plt.subplots()
    plt.figure(figsize=(3.5, 2.5))
    s_po_flag = is_pareto_efficient(stage)
    lat_s, cost_s = stage[:, 0], stage[:, 1]

    s_pareto = stage[s_po_flag]
    lat_po_s, cost_po_s = s_pareto[:, 0][np.argsort(s_pareto[:, 0])], s_pareto[:, 1][np.argsort(s_pareto[:, 0])]

    plt.scatter(cost_s, lat_s, marker=m, color=c, label=f"Dominated")
    plt.plot(cost_po_s, lat_po_s, marker=">", color=c, label=f"Pareto")
    for j, (cost, l) in enumerate(zip(cost_s, lat_s)):
        plt.annotate(f"{j + 1}", (cost, l))

    plt.xlabel("SubQ Cost ($)")
    plt.ylabel("SubQ Latency (s)")
    # ax.set_ylabel('Latency', fontdict={"size": 20})
    # ax.set_xlabel('Cost', fontdict={"size": 20})
    # plt.xticks(fontsize=20)
    # plt.yticks(fontsize=20)
    # ax.set_ylabel('Latency')
    # ax.set_xlabel('Cost')
    # ax.set_title(f"Query Stage {i + 1}", fontdict={"size": 20})
    # ax.legend(fontsize=20)
    plt.legend()
    plt.tight_layout()
    # plt.show()
    if save_fig:
        plot_path = f"./output/final_expts/plots/updated_eg_missing_points"
        if not os.path.exists(plot_path):
            os.makedirs(plot_path, exist_ok=True)
        plt.savefig(f"{plot_path}/subQ_{i + 1}.pdf")

def plot_query(query, save_fig):
    # fig, ax = plt.subplots()
    # lat_q, cost_q = query[:, 0], query[:, 1]
    # ax.scatter(cost_q, lat_q, marker=".", color="green", label="Dominated")
    #
    # q_pareto_flag = is_pareto_efficient(query)
    # q_pareto = query[q_pareto_flag]
    # lat_q_pareto, cost_q_pareto = q_pareto[:, 0][np.argsort(q_pareto[:, 0])], q_pareto[:, 1][np.argsort(q_pareto[:, 0])]
    # ax.plot(cost_q_pareto, lat_q_pareto, marker=">", color="green", label="Pareto")
    # for j, (cost, l) in enumerate(zip(cost_q, lat_q)):
    #     ax.annotate(f"{j + 1}", (cost, l), size=20)
    #
    # # ax.set_ylabel('Latency')
    # # ax.set_xlabel('Cost')
    # ax.set_title(f"Query-level Solutions", fontdict={"size": 20})
    # ax.set_ylabel('Latency', fontdict={"size": 20})
    # ax.set_xlabel('Cost', fontdict={"size": 20})
    # plt.xticks(fontsize=20)
    # plt.yticks(fontsize=20)
    # ax.legend(fontsize=20)
    # plt.tight_layout()
    # plt.show()

    plt.figure(figsize=(3.5, 2.5))
    lat_q, cost_q = query[:, 0], query[:, 1]
    plt.scatter(cost_q, lat_q, marker=".", color="green", label="Dominated")

    q_pareto_flag = is_pareto_efficient(query)
    q_pareto = query[q_pareto_flag]
    lat_q_pareto, cost_q_pareto = q_pareto[:, 0][np.argsort(q_pareto[:, 0])], q_pareto[:, 1][np.argsort(q_pareto[:, 0])]
    plt.plot(cost_q_pareto, lat_q_pareto, marker=">", color="green", label="Pareto")
    for j, (cost, l) in enumerate(zip(cost_q, lat_q)):
        plt.annotate(f"{j + 1}", (cost, l))

    # ax.set_ylabel('Latency')
    # ax.set_xlabel('Cost')
    plt.ylabel('Query Latency (s)')
    plt.xlabel('Cloud Cost ($)')
    plt.legend()
    plt.tight_layout()
    # plt.show()
    if save_fig:
        plot_path = f"./output/final_expts/plots/updated_eg_missing_points"
        if not os.path.exists(plot_path):
            os.makedirs(plot_path, exist_ok=True)
        plt.savefig(f"{plot_path}/query.pdf")

def cal_query(s_list):
    query = np.zeros_like(s_list[0])
    # serial stages
    for s in s_list:
        query = query + s

    return query

def a_test():
    # plot
    n_samples = 1000
    seed = 0
    np.random.seed(seed)
    latency = np.random.random((n_samples, 1)) * 5
    cost = np.random.random((n_samples, 1)) * 5
    query = np.hstack((latency, cost))

    fig, ax = plt.subplots()

    ax.scatter(cost, latency, marker="o", color="grey", label="Dominated")

    q_pareto_flag = is_pareto_efficient(query)
    q_pareto = query[q_pareto_flag]
    lat_q_pareto, cost_q_pareto = q_pareto[:, 0], q_pareto[:, 1]
    ax.scatter(cost_q_pareto, lat_q_pareto, marker="o", color="blue", label="Pareto")

    ax.set_ylabel('Query Latency', fontdict={"size": 20})
    ax.set_xlabel('Cloud Cost', fontdict={"size": 20})
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    ax.legend(fontsize=20)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 2-dominance dominates the local optimal
    # stage1 = np.array([[9, 18.5], [8, 18], [7.5, 20.5], [7, 24], [5, 20], [5.1, 21.1], [5, 21], [5, 22]])
    # stage2 = np.array([[10, 18.2], [7, 19.3], [6.5, 23.8], [9, 18], [5.9, 20.6], [5.5, 21.9], [6.0, 21], [5, 24]])
    # stage3 = np.array([[9.5, 17.5], [8.5, 17], [7, 19.7], [6.3, 18], [5.1, 22], [3.8, 20.3], [4, 20.1], [3.5, 20]])

    # # original with missing solutions, with index 1, 3, 6, 5, 7 (start from 1 rather than 0)
    stage1 = np.array([[8, 18], [7.5, 20.5], [7, 24], [5, 20], [5.1, 21.1], [5, 21], [5, 22]])
    stage2 = np.array([[7, 19.3], [6.5, 23.8], [9, 18], [5.9, 20.6], [5.5, 21.9], [6.0, 21], [5, 24]])
    stage3 = np.array([[8.5, 17], [7, 19.7], [6.3, 18], [5.1, 22], [3.8, 20.3], [4, 20.1], [3.5, 20]])

    # # other missing solutions (stage 3 has solutions located in the left-upper corner)
    # stage1 = np.array([[6, 20], [8, 18], [7.5, 20.5], [7, 24], [5, 20], [5.1, 21.1], [5, 21], [5, 22]])
    # stage2 = np.array([[7.2, 19.5], [7, 19.3], [6.5, 23.8], [9, 18], [5.9, 20.6], [5.5, 21.9], [6.0, 21], [5, 24]])
    # stage3 = np.array([[9, 18], [8.5, 17], [7, 19.7], [6.3, 18], [5.1, 22], [3.8, 20.3], [4, 20.1], [3.5, 20]])

    # # other missing solutions (stage 1,3 has solutions located in the left-upper corner, stage 2 has a solutions in right-bottom corner)
    # stage1 = np.array([[8.2, 18.5], [8, 18], [7.5, 20.5], [7, 24], [5, 20], [5.1, 21.1], [5, 21], [5, 22]])
    # stage2 = np.array([[5.1, 24.2], [7, 19.3], [6.5, 23.8], [9, 18], [5.9, 20.6], [5.5, 21.9], [6.0, 21], [5, 24]])
    # stage3 = np.array([[8.7, 17.2], [8.5, 17], [7, 19.7], [6.3, 18], [5.1, 22], [3.8, 20.3], [4, 20.1], [3.5, 20]])

    # other missing solutions (stage 3 has solutions located in the left-upper corner)
    # stage1 = np.array([[8.1, 18.5], [8, 18], [7.5, 20.5], [7, 24], [5, 20], [5.1, 21.1], [5, 21], [5, 22]])
    # stage2 = np.array([[7.1, 20], [7, 19.3], [6.5, 23.8], [9, 18], [5.9, 20.6], [5.5, 21.9], [6.0, 21], [5, 24]])
    # stage3 = np.array([[8.7, 20.5], [8.5, 17], [7, 19.7], [6.3, 18], [5.1, 22], [3.8, 20.3], [4, 20.1], [3.5, 20]])

    s_list = [stage1, stage2, stage3]
    colors = ["red", "blue", "orange"]
    markers = [".", ".", "."]

    for i, (s, m, c) in enumerate(zip(s_list, markers, colors)):
        plot_stage(i, s, m, c, save_fig=True)

    query = cal_query(s_list)
    plot_query(query, save_fig=True)
    # test()