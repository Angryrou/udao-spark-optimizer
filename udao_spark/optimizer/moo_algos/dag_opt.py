# Copyright (c) 2024 École Polytechnique
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Author(s): Qi FAN <qi dot fan at polytechnique dot edu>
#
# Description: DAG optimization in query_moo_general (DAG is degenerated as list)
#
# Created at 04/01/2024
import itertools
import time
from multiprocessing import Pool
from typing import Any, List, Tuple, Union

import numpy as np
import torch as th
from tqdm import tqdm  # type: ignore

from udao_spark.optimizer.utils import is_pareto_efficient


class DAGOpt:
    def __init__(
        self,
        f: np.ndarray,
        conf: Union[th.Tensor, np.ndarray],
        indices_arr: np.ndarray,
        algo: str,
        runmode: str,
    ) -> None:
        self.f = f
        self.conf = conf
        self.indices_arr = indices_arr
        # hier_moo; seq_div_and_conq; approx_solve
        self.algo = algo
        self.runmode = runmode
        self.verbose = False

        self.len_theta = conf.shape[1]
        self.n_objs = f.shape[1]

    def solve(self) -> Tuple[np.ndarray, np.ndarray]:
        if "hier_moo" in self.algo:
            # weights
            n_ws = int(self.algo.split("%")[1])
            ws_steps = 1 / (int(n_ws) - 1)
            ws_pairs = self.even_weights(ws_steps, self.n_objs)
            F, Theta = self._hier_moo(self.f, self.conf, ws_pairs, self.indices_arr)
        elif self.algo == "GD":
            F, Theta = self._seq_div_and_conq(
                self.f,
                self.conf,
                self.indices_arr,
            )
        elif self.algo == "B":
            F, Theta = self.approx_solve(self.f, self.conf, self.indices_arr)
        else:
            raise Exception(f"mode {self.algo} is not supported in {classmethod}!")

        return F, Theta

    # ------------Approximate solve------------------------------#
    def approx_solve(
        self,
        f_th: np.ndarray,
        conf_th: Union[th.Tensor, np.ndarray],
        indices_arr: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        start = time.time()
        sorted_index = np.lexsort(
            (indices_arr[:, 2], indices_arr[:, 1], indices_arr[:, 0])
        )
        qs_indices = indices_arr[sorted_index]
        stages = np.unique(qs_indices[:, 0])
        np.unique(qs_indices[:, 1]).astype(int)
        if isinstance(conf_th, np.ndarray):
            # qs_f_all = f_th[sorted_index]
            qs_conf_all = conf_th[sorted_index]
        else:
            assert isinstance(conf_th, th.Tensor)
            # qs_f_all = f_th.numpy()[sorted_index]
            qs_conf_all = conf_th.numpy()[sorted_index]
        qs_f_all = f_th[sorted_index]
        uniq_theta_c = np.unique(qs_conf_all[:, :8], axis=0)
        print(f"the number of theta_c is: {uniq_theta_c.shape[0]}")
        if self.verbose:
            print(
                f"time cost of getting values and indices of "
                f"f and conf is {time.time() - start}"
            )
        qs_f_all_list = [qs_f_all[np.where(qs_indices[:, 0] == i)] for i in stages]
        qs_conf_all_list = [
            qs_conf_all[np.where(qs_indices[:, 0] == i)] for i in stages
        ]

        boundary_f_list = []
        boundary_conf_list = []
        for qs_id, qs_f, qs_conf in zip(stages, qs_f_all_list, qs_conf_all_list):
            time.time()
            len_p_counts = np.unique(
                qs_indices[np.where(qs_indices[:, 0] == qs_id)][:, 1],
                return_counts=True,
            )[1]
            cumsum_counts = np.cumsum(len_p_counts)[:-1]
            all_f_values = np.split(qs_f, cumsum_counts)
            all_conf_values = np.split(qs_conf, cumsum_counts)
            p_max = len_p_counts.max()
            padded_f_arr = np.array(
                [
                    np.tile(array, (int(p_max / array.shape[0] + 1), 1))[:p_max, :]
                    if array.shape[0] < p_max
                    else array
                    for array in all_f_values
                ]
            )
            min_row_inds_per_obj = np.argmin(padded_f_arr, axis=1)
            min_rows_f = padded_f_arr[
                np.arange(padded_f_arr.shape[0])[:, None], min_row_inds_per_obj, :
            ].reshape(-1, self.n_objs)
            boundary_f_list.append(min_rows_f)

            min_conf = [
                conf[min_ind].tolist()
                for conf, min_ind in zip(all_conf_values, min_row_inds_per_obj)
            ]

            confs = np.array(sum(min_conf, []))
            assert confs.shape[0] == min_rows_f.shape[0]
            boundary_conf_list.append(confs)

        assert len(boundary_f_list) == len(boundary_conf_list)
        if self.verbose:
            print(f"time cost of getting boundary points is: {time.time() - start}")
        all_f_qs = np.hstack(boundary_f_list)
        all_confs_qs = np.hstack(boundary_conf_list)
        assert all_confs_qs.shape[1] == self.len_theta * len(stages)

        all_query_lat = np.sum(all_f_qs[:, 0::2], axis=1)
        all_query_cost = np.sum(all_f_qs[:, 1::2], axis=1)
        boundary_query_objs = np.vstack((all_query_lat, all_query_cost)).T

        if self.verbose:
            print(
                f"time cost of getting boundary query-level objs "
                f"is {time.time() - start}"
            )

        start_filter_global = time.time()
        po_query_ind = is_pareto_efficient(boundary_query_objs)
        po_query_objs = boundary_query_objs[po_query_ind]
        po_query_confs = all_confs_qs[po_query_ind]
        uniq_po_query_objs, uniq_po_query_inds = np.unique(
            po_query_objs, axis=0, return_index=True
        )
        uniq_po_query_confs = po_query_confs[uniq_po_query_inds]

        if self.verbose:
            print(
                f"time cost of filtering globally is"
                f" {time.time() - start_filter_global}"
            )
            print(f"Pareto frontier is {uniq_po_query_objs}")

        return uniq_po_query_objs, uniq_po_query_confs

    # -------------General Hierarchical MOO-----------------------#
    def _hier_moo(
        self,
        f_th: np.ndarray,
        conf_th: Union[th.Tensor, np.ndarray],
        ws_pairs: List[List[Any]],
        indices_arr: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        start = time.time()
        sorted_index = np.lexsort(
            (indices_arr[:, 2], indices_arr[:, 1], indices_arr[:, 0])
        )
        qs_indices = indices_arr[sorted_index]
        stages = np.unique(qs_indices[:, 0])
        c_ids = np.unique(qs_indices[:, 1]).astype(int)
        print(f"the number of theta_c is: {c_ids.shape[0]}")
        # qs_f_all = f_th.numpy()[sorted_index]
        # qs_conf_all = conf_th.numpy()[sorted_index]
        if isinstance(conf_th, np.ndarray):
            # qs_f_all = f_th[sorted_index]
            qs_conf_all = conf_th[sorted_index]
        else:
            assert isinstance(conf_th, th.Tensor)
            # qs_f_all = f_th.numpy()[sorted_index]
            qs_conf_all = conf_th.numpy()[sorted_index]
        qs_f_all = f_th[sorted_index]
        if self.verbose:
            print(
                f"time cost of getting values and indices of f and conf "
                f"is {time.time() - start}"
            )

        qs_f_all_list = [qs_f_all[np.where(qs_indices[:, 0] == i)] for i in stages]
        qs_conf_all_list = [
            qs_conf_all[np.where(qs_indices[:, 0] == i)] for i in stages
        ]

        qs_c_f_list = []
        qs_c_conf_list = []
        for qs_id, qs_f, qs_conf in zip(stages, qs_f_all_list, qs_conf_all_list):
            len_p_counts = np.unique(
                qs_indices[np.where(qs_indices[:, 0] == qs_id)][:, 1],
                return_counts=True,
            )[1]
            cumsum_counts = np.cumsum(len_p_counts)[:-1]
            all_f_values = np.split(qs_f, cumsum_counts)
            all_conf_values = np.split(qs_conf, cumsum_counts)
            assert len(all_f_values) == c_ids.shape[0]
            assert len(all_conf_values) == c_ids.shape[0]

            qs_c_f_list.append(all_f_values)
            qs_c_conf_list.append(all_conf_values)

        if c_ids.shape[0] > 500:
            mode = "multiprocessing"
        else:
            mode = "naive"

        if mode == "multiprocessing":
            arg_list = [
                (ws, c_id, stages, qs_c_f_list, qs_c_conf_list)
                for c_id in tqdm(c_ids, total=c_ids.shape[0])
                for ws in ws_pairs
            ]

            with Pool(processes=10) as pool:
                ret_list = pool.starmap(self._ws_per_c_all_nodes, arg_list)

            query_f_list = [result[0] for result in ret_list]
            query_conf_list = [result[1] for result in ret_list]
        else:
            query_f_list = []
            query_conf_list = []
            for c_id in tqdm(c_ids, total=c_ids.shape[0]):
                for ws in ws_pairs:
                    objs, confs = self._ws_per_c_all_nodes(
                        ws, c_id, stages, qs_c_f_list, qs_c_conf_list
                    )
                    query_f_list.append(objs)
                    query_conf_list.append(confs)

        start_filter_global = time.time()
        query_f_arr = np.concatenate(query_f_list)
        query_conf_arr = np.concatenate(query_conf_list)
        po_query_ind = is_pareto_efficient(query_f_arr)
        po_query_objs = query_f_arr[po_query_ind]
        po_query_confs = query_conf_arr[po_query_ind]
        if self.verbose:
            print(
                f"time cost of filtering globally "
                f"is {time.time() - start_filter_global}"
            )

        uniq_po_query_objs, uniq_po_query_inds = np.unique(
            po_query_objs, axis=0, return_index=True
        )
        uniq_po_query_confs = po_query_confs[uniq_po_query_inds]
        print(f"Pareto frontier is {uniq_po_query_objs}")
        return uniq_po_query_objs, uniq_po_query_confs

    def _ws_per_c_all_nodes(
        self,
        ws: List[Any],
        c_id: int,
        stages: np.ndarray,
        qs_c_f_list: List[Any],
        qs_c_conf_list: List[Any],
    ) -> Tuple[np.ndarray, np.ndarray]:
        sum_objs = np.zeros(
            [
                self.n_objs,
            ]
        )
        sel_conf_list = []
        for qs_id, qs_f_cid_list, qs_conf_cid_list in zip(
            stages, qs_c_f_list, qs_c_conf_list
        ):
            objs = qs_f_cid_list[c_id].reshape(-1, self.n_objs)
            confs = qs_conf_cid_list[c_id].reshape(-1, self.len_theta)
            objs_min, objs_max = objs.min(0), objs.max(0)
            if all((objs_min - objs_max) <= 0):
                obj = np.sum(objs * ws, axis=1)
                po_ind = int(np.argmin(obj))
                sum_objs += objs[po_ind]

                sel_conf = confs[po_ind].reshape(-1, self.len_theta)
                sel_conf_list.append(sel_conf)
            else:
                raise Exception(
                    "Cannot do normalization! "
                    "Lower bounds of objective values "
                    "are higher than its upper bounds."
                )

        all_qs_confs = np.hstack(sel_conf_list)
        assert all_qs_confs.shape[1] == stages.shape[0] * self.len_theta
        return sum_objs.reshape(-1, self.n_objs), all_qs_confs

    # -------------Sequential Divide-and-Conquer-------------------#
    def _seq_div_and_conq(
        self,
        f_th: np.ndarray,
        conf_th: Union[th.Tensor, np.ndarray],
        indices_arr: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.verbose:
            print("FUNCTION: Seq_div_and_conq starts:")
            print()
        start = time.time()
        sorted_index = np.lexsort(
            (indices_arr[:, 2], indices_arr[:, 1], indices_arr[:, 0])
        )
        qs_indices = indices_arr[sorted_index]
        stages = np.unique(qs_indices[:, 0])
        c_ids = np.unique(qs_indices[:, 1]).astype(int)
        # print(f"the number of theta_c is: {c_ids.shape[0]}")
        # if isinstance(f_th, np.ndarray):
        #     qs_f_all = f_th[sorted_index]
        #     qs_conf_all = conf_th[sorted_index]
        # else:
        #     assert isinstance(f_th, th.Tensor)
        #     qs_f_all = f_th.numpy()[sorted_index]
        #     qs_conf_all = conf_th.numpy()[sorted_index]

        if isinstance(conf_th, np.ndarray):
            # qs_f_all = f_th[sorted_index]
            qs_conf_all = conf_th[sorted_index]
        else:
            assert isinstance(conf_th, th.Tensor)
            # qs_f_all = f_th.numpy()[sorted_index]
            qs_conf_all = conf_th.numpy()[sorted_index]
        qs_f_all = f_th[sorted_index]
        uniq_theta_c = np.unique(qs_conf_all[:, :8], axis=0)
        print(f"the number of theta_c is: {uniq_theta_c.shape[0]}")
        if self.verbose:
            print(
                f"time cost of getting values and indices of f and conf "
                f"is {time.time() - start}"
            )

        qs_f_all_list = [qs_f_all[np.where(qs_indices[:, 0] == i)] for i in stages]
        qs_conf_all_list = [
            qs_conf_all[np.where(qs_indices[:, 0] == i)] for i in stages
        ]

        qs_c_f_list = []
        qs_c_conf_list = []
        for qs_id, qs_f, qs_conf in zip(stages, qs_f_all_list, qs_conf_all_list):
            len_p_counts = np.unique(
                qs_indices[np.where(qs_indices[:, 0] == qs_id)][:, 1],
                return_counts=True,
            )[1]
            cumsum_counts = np.cumsum(len_p_counts)[:-1]
            all_f_values = np.split(qs_f, cumsum_counts)
            all_conf_values = np.split(qs_conf, cumsum_counts)
            assert len(all_f_values) == c_ids.shape[0]
            assert len(all_conf_values) == c_ids.shape[0]

            qs_c_f_list.append(all_f_values)
            qs_c_conf_list.append(all_conf_values)

        if c_ids.shape[0] > 500:
            runmode = "multiprocessing"
        else:
            runmode = "naive"

        if runmode == "multiprocessing":
            arg_list = [
                (c_id, stages, qs_c_f_list, qs_c_conf_list)
                for c_id in tqdm(c_ids, total=c_ids.shape[0])
            ]
            with Pool(processes=2) as pool:
                ret_list = pool.starmap(self._div_and_conq_non_recur, arg_list)

            query_f_list = [result[0].tolist() for result in ret_list]
            query_conf_list = [result[1].tolist() for result in ret_list]
        else:
            query_f_list = []
            query_conf_list = []
            for c_id in tqdm(c_ids, total=c_ids.shape[0]):
                objs, confs = self._div_and_conq_non_recur(
                    c_id, stages, qs_c_f_list, qs_c_conf_list
                )
                query_f_list.append(objs)
                query_conf_list.append(confs)

        start_filter_global = time.time()
        query_f_arr = np.concatenate(query_f_list)
        query_conf_arr = np.concatenate(query_conf_list)
        po_query_ind = is_pareto_efficient(query_f_arr)
        po_query_objs = query_f_arr[po_query_ind]
        po_query_confs = query_conf_arr[po_query_ind]

        if self.verbose:
            print(
                f"time cost of filtering globally "
                f"is {time.time() - start_filter_global}"
            )

        uniq_po_query_objs, uniq_po_query_inds = np.unique(
            po_query_objs, axis=0, return_index=True
        )
        uniq_po_query_confs = po_query_confs[uniq_po_query_inds]
        print(f"Pareto frontier is {uniq_po_query_objs}")
        return uniq_po_query_objs, uniq_po_query_confs

    def _merge(
        self,
        node1: Tuple[np.ndarray, np.ndarray],
        node2: Tuple[np.ndarray, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        all_objs, all_confs = self._compute_all_configurations(node1, node2)
        po_objs, po_confs = self._compute_pareto_frontier_efficient(all_objs, all_confs)
        return po_objs, po_confs

    def _div_and_conq_non_recur(
        self,
        c_id: int,
        stages: np.ndarray,
        qs_c_f_list: List[Any],
        qs_c_conf_list: List[Any],
    ) -> Tuple[np.ndarray, np.ndarray]:
        stages_str = [f"qs{int(qs_id)}" for qs_id in stages]
        stack_f = [qs_f[c_id] for qs_f in qs_c_f_list.copy()]
        stack_conf = [qs_conf[c_id] for qs_conf in qs_c_conf_list.copy()]
        while len(stages_str) > 1:
            if len(stages_str) == 2 and len(stages_str) == 2:
                node1_qs_id = stages_str[0]
                node2_qs_id = stages_str[1]
                node1 = (stack_f[0], stack_conf[0])
                node2 = (stack_f[1], stack_conf[1])

                compressed_f, compressed_conf = self._merge(node1, node2)
                stages_str.append(f"{node1_qs_id}+{node2_qs_id}")
                stages_str.remove(node1_qs_id)
                stages_str.remove(node2_qs_id)
                assert len(stages_str) == 1

                if "+" in node1_qs_id:
                    node1_stages_inds = [
                        int(x.split("qs")[1]) for x in node1_qs_id.split("+")
                    ]
                else:
                    node1_stages_inds = [int(node1_qs_id.split("qs")[1])]

                if "+" in node2_qs_id:
                    node2_stages_inds = [
                        int(x.split("qs")[1]) for x in node2_qs_id.split("+")
                    ]
                else:
                    node2_stages_inds = [int(node2_qs_id.split("qs")[1])]

                stages_inds = node1_stages_inds + node2_stages_inds
                node1_confs = node1[1][compressed_conf[:, 0]]
                node2_confs = node2[1][compressed_conf[:, 1]]
                confs = np.hstack((node1_confs, node2_confs))
                assert confs.shape[0] == compressed_f.shape[0]
                split_confs = np.hsplit(confs, len(stages_inds))

                sort_inds = np.argsort(stages_inds)
                # let confs follow the order of qs0, 1, ...
                final_confs = np.hstack([split_confs[x] for x in sort_inds])

                assert final_confs.shape[0] == compressed_f.shape[0]
                assert final_confs.shape[1] == self.len_theta * len(stages_inds)

                result = (compressed_f, final_confs)
                break

            else:
                # get sub_problems
                n_sub_problems = (
                    int(len(stages_str) / 2)
                    if len(stages_str) % 2 == 0
                    else int(len(stages_str) / 2) + 1
                )
                tmp_stages = stages_str.copy()
                new_stack_f_list: List[np.ndarray] = []
                new_stack_conf_list: List[np.ndarray] = []
                for i in range(n_sub_problems):
                    if i * 2 + 1 == len(tmp_stages):
                        new_stack_f_list.insert(0, stack_f[-1])
                        new_stack_conf_list.insert(0, stack_conf[-1])
                    else:
                        node1_qs_id = tmp_stages[i * 2]
                        node2_qs_id = tmp_stages[i * 2 + 1]

                        node1 = (
                            stack_f[i * 2].reshape(-1, 2),
                            stack_conf[i * 2],
                        )
                        node2 = (
                            stack_f[i * 2 + 1].reshape(-1, 2),
                            stack_conf[i * 2 + 1],
                        )
                        compressed_f, compressed_conf = self._merge(node1, node2)

                        new_stack_f_list.append(compressed_f)

                        stages_str.append(f"{node1_qs_id}+{node2_qs_id}")
                        stages_str.remove(node1_qs_id)
                        stages_str.remove(node2_qs_id)

                        node1_confs = node1[1][compressed_conf[:, 0]]
                        node2_confs = node2[1][compressed_conf[:, 1]]

                        confs = np.hstack((node1_confs, node2_confs))
                        assert confs.shape[0] == compressed_f.shape[0]
                        new_stack_conf_list.append(confs)

                stack_f = new_stack_f_list
                stack_conf = new_stack_conf_list
                assert len(stack_f) == len(stack_conf)

        return result

    def _compute_pareto_frontier_efficient(
        self,
        all_objs: List[List[Any]],
        all_confs: List[Tuple[Any, ...]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        arr_points = np.array(all_objs)
        po_inds = is_pareto_efficient(arr_points).tolist()
        po_objs = arr_points[po_inds]

        po_confs = np.array(all_confs)[po_inds]

        return po_objs, po_confs

    def _compute_all_configurations(
        self,
        node1: Tuple[np.ndarray, np.ndarray],
        node2: Tuple[np.ndarray, np.ndarray],
    ) -> Tuple[List[List[Any]], List[Tuple[Any, ...]]]:
        node1_f = np.unique(node1[0], axis=0)
        node2_f = np.unique(node2[0], axis=0)
        pre_product = [list(range(node1_f.shape[0])), list(range(node2_f.shape[0]))]

        def mass_evaluate(raw_conf: Tuple[Any, ...]) -> List[Any]:
            i, j = raw_conf[0], raw_conf[1]
            latency, cost = np.sum([node1_f[i], node2_f[j]], axis=0, dtype=np.float64)
            return [latency, cost]

        all_confs = list(itertools.product(*pre_product))
        objs = list(map(mass_evaluate, all_confs))
        confs = all_confs

        return objs, confs

    # generate even weights for 2d and 3D
    def even_weights(self, stepsize: Any, m: int) -> List[List[Any]]:
        if m == 2:
            w1 = np.hstack([np.arange(0, 1, stepsize), 1])
            w2 = 1 - w1
            ws_pairs = [[w1, w2] for w1, w2 in zip(w1, w2)]

        elif m == 3:
            w_steps = np.linspace(0, 1, num=int(1 / stepsize) + 1, endpoint=True)
            for i, w in enumerate(w_steps):
                # use round to avoid case of floating point limitations in Python
                # the limitation: 1- 0.9 = 0.09999999999998 rather than 0.1
                other_ws_range = round((1 - w), 10)
                w2 = np.linspace(
                    0,
                    other_ws_range,
                    num=round(other_ws_range / stepsize + 1),
                    endpoint=True,
                )
                w3 = other_ws_range - w2
                num = w2.shape[0]
                w1 = np.array([w] * num)
                ws = np.hstack(
                    [w1.reshape([num, 1]), w2.reshape([num, 1]), w3.reshape([num, 1])]
                )
                if i == 0:
                    ws_pairs = ws.tolist()
                else:
                    ws_pairs = np.vstack([ws_pairs, ws]).tolist()
        else:
            raise Exception("Current it only supports 2D and 3D!")

        assert all(np.round(np.sum(np.array(ws_pairs), axis=1), 10) == 1)
        return ws_pairs
