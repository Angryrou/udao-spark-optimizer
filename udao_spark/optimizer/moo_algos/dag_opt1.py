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
import sys

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
        if "WS" in self.algo:
            # weights
            n_ws = int(self.algo.split("&")[1])
            ws_steps = 1 / (int(n_ws) - 1)
            ws_pairs = self.even_weights(ws_steps, self.n_objs)
            F1, Theta1 = self._hier_moo_all_c(self.f, self.conf, ws_pairs, self.indices_arr)
            F, Theta = self._hier_moo(self.f, self.conf, ws_pairs, self.indices_arr)
            assert np.all(F == F1)
            assert np.all(Theta == Theta1)
        elif self.algo == "GD":
            F, Theta = self._seq_div_and_conq(
                self.f,
                self.conf,
                self.indices_arr,
            )
            F1, Theta1 = self._seq_div_and_conq_all_c(self.f,
                self.conf,
                self.indices_arr,)
            assert np.all(F == F1)
            assert np.all(Theta == Theta1)
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
            # time.time()
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

    def _hier_moo_all_c(
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

        query_f_list = []
        query_conf_list = []
        for node_id in stages:
            qs_c_f_node = qs_c_f_list[int(node_id)]
            qs_c_conf_node = qs_c_conf_list[int(node_id)]

            objs, confs = self._ws_per_node_all_c(ws_pairs,
                                                  qs_c_f_node,
                                                  qs_c_conf_node)
            query_f_list.append(objs)
            query_conf_list.append(confs)

        # sum all nodes: shape (n_theta_c, n_ws * n_objs),
        # where 0 (even id) is lat, 1 (odd id) is cost
        query_values_all_ws = np.sum(np.array(query_f_list), axis=0)
        # query_f_arr: shape (n_ws * n_theta_c, n_objs)
        # order: under each weight, following n_theta_c solutions, i.e. n_theta_c_ws_1 solutions, ..., n_theta_c_ws_i,...
        query_f_arr = np.vstack(np.split(query_values_all_ws, len(ws_pairs), axis=1))
        # query_confs_all_ws: a list with length of n_stages
        # each item in the list is an array with shape (n_ws * n_theta_c, n_theta)
        # order of each array: under each weight, following n_theta_c solutions
        query_confs_all_ws = [np.vstack(np.split(x, len(ws_pairs), axis=1))
                              for x in query_conf_list]
        start_filter_global = time.time()
        po_query_ind = is_pareto_efficient(query_f_arr)
        po_query_objs = query_f_arr[po_query_ind]
        # fixme: confs not the same
        po_query_confs = np.hstack([x[po_query_ind] for x in query_confs_all_ws])
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
            objs_norm = (objs - objs_min) / (objs_max - objs_min)
            if all((objs_min - objs_max) <= 0):
                obj = np.sum(objs_norm * ws, axis=1)
                # obj = np.sum(objs * ws, axis=1)
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

    def _ws_per_node_all_c(
            self,
            ws_pairs: List[Any],
            qs_c_f_node: List[Any],
            qs_c_conf_node: List[Any],
    ) -> Tuple[np.ndarray, np.ndarray]:

        # the objective values of all theta_c,
        # a list with n_theta_c items, each item is an array with shape (n_solutions, 2)
        # qs_c_f_node = qs_c_f_list[int(node_id)]
        # qs_c_conf_node = qs_c_conf_list[int(node_id)]
        n_theta_c = len(qs_c_f_node)
        assert n_theta_c == len(qs_c_conf_node)

        # the maximum number of solutions among all theta_c (n_max)
        n_max = max([x.shape[0] for x in qs_c_f_node])

        # under each theta_c, apply WS for all solutions within the node
        # 1. normalize objs
        # 2. norm_objs * ws_pairs_arr = shape (2 * n_solutions) \times shape (n_ws * 2) = shape (n_solutions * n_ws)
        # 3. find the index of the min_ws of step 2 under each weight: output --> shape (1, n_ws)

        # considering multiple theta_c, where under each theta_c, n_solutions could differ
        # --> find the n_max, and pad the solutions array of the node to shape (n_theta_c, n_max)
        # output shape should be (n_theta_c, n_ws)

        # 1. normalize objs of all theta_c
        ## pad the solution array with n_max by the value -1
        n_diff_to_n_max = [n_max - x.shape[0] for x in qs_c_f_node]
        qs_c_f_node_enrich_list = []
        qs_c_conf_node_enrich_list = []
        for objs, confs, n_diff in zip(qs_c_f_node, qs_c_conf_node, n_diff_to_n_max):
            # set the extension with two different and large values,
            # to make sure 1) max-min != 0 and 2) won't be selected after ws (to minimize)
            qs_c_f_node_enrich_list.extend(objs.tolist() + [[sys.maxsize, sys.maxsize - 1]] * n_diff)
            qs_c_conf_node_enrich_list.extend(confs.tolist() + [[-1] * self.len_theta] * n_diff)
        qs_c_f_node_enrich_arr = np.array(qs_c_f_node_enrich_list)
        qs_c_conf_node_enrich_arr = np.array(qs_c_conf_node_enrich_list)

        assert qs_c_f_node_enrich_arr.shape[0] == n_theta_c * n_max
        assert qs_c_f_node_enrich_arr.shape[0] == qs_c_conf_node_enrich_arr.shape[0]

        ## find the min and max of objectives
        objs_min = np.array([x.min(0) for x in qs_c_f_node])
        objs_max = np.array([x.max(0) for x in qs_c_f_node])
        assert np.all(objs_max - objs_min >= 0)

        ## pad them with n_max
        objs_min_repeat = np.repeat(objs_min, n_max, axis=0)
        objs_max_repeat = np.repeat(objs_max, n_max, axis=0)
        assert objs_max_repeat.shape[0] == n_theta_c * n_max
        assert objs_max_repeat.shape[0] == objs_min_repeat.shape[0]


        norm_objs = (qs_c_f_node_enrich_arr - objs_min_repeat) / (objs_max_repeat - objs_min_repeat)
        # the case with only one solution under one theta_c
        inds_one_solution = np.where(~(objs_max_repeat - objs_min_repeat).any(axis=1))[0]
        # set them to be 0 as it won't affect the ws values
        norm_objs[inds_one_solution] = 0
        # 2. norm_objs * ws_pairs_arr
        # shape (n_ws, 2)
        ws_pairs_arr = np.array(ws_pairs)
        # shape (n_theta_c*n_max, n_ws)
        ws_objs = np.matmul(norm_objs, ws_pairs_arr.T)



        # 3. find the index of min_ws, shape (n_theta_c, n_ws)
        # inds_select = np.zeros([n_theta_c, ws_pairs_arr.shape[0]])
        split_ws_objs = np.split(ws_objs, int(ws_objs.shape[0] / n_max))
        inds_select = [np.argmin(x, axis=0) for x in split_ws_objs]

        # return selected solution among all weights (n_theta_c, n_ws * n_objs)
        qs_f_node_select = np.vstack([np.hstack([x[ind] for ind in inds_all_ws])
                                      for x, inds_all_ws in zip(qs_c_f_node, inds_select)])
        qs_conf_node_select = np.vstack([np.hstack([x[ind] for ind in inds_all_ws])
                                         for x, inds_all_ws in zip(qs_c_conf_node, inds_select)])

        return qs_f_node_select, qs_conf_node_select

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

    def _seq_div_and_conq_all_c(
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

        if isinstance(conf_th, np.ndarray):
            qs_conf_all = conf_th[sorted_index]
        else:
            assert isinstance(conf_th, th.Tensor)
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

        query_f_list, query_conf_list = self._div_and_conq_non_recur_all_c(stages, qs_c_f_list, qs_c_conf_list)

        query_f_list1 = []
        query_conf_list1 = []
        for c_id in tqdm(c_ids, total=c_ids.shape[0]):
            objs1, confs1 = self._div_and_conq_non_recur(
                c_id, stages, qs_c_f_list, qs_c_conf_list
            )
            query_f_list1.append(objs1)
            query_conf_list1.append(confs1)

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

    def _merge_all_c(
        self,
        node1: Tuple[np.ndarray, np.ndarray],
        node2: Tuple[np.ndarray, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        all_objs, all_confs = self._compute_all_configurations_all_c(node1, node2)
        po_objs, po_confs = self._compute_pareto_frontier_efficient_all_c(all_objs, all_confs)
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

    def _div_and_conq_non_recur_all_c(
        self,
        stages: np.ndarray,
        qs_c_f_list: List[Any],
        qs_c_conf_list: List[Any],
    ) -> Tuple[np.ndarray, np.ndarray]:
        stages_str = [f"qs{int(qs_id)}" for qs_id in stages]
        stack_f = [qs_f for qs_f in qs_c_f_list.copy()]
        stack_conf = [qs_conf  for qs_conf in qs_c_conf_list.copy()]
        while len(stages_str) > 1:
            if len(stages_str) == 2 and len(stages_str) == 2:
                node1_qs_id = stages_str[0]
                node2_qs_id = stages_str[1]
                node1 = (stack_f[0], stack_conf[0])
                node2 = (stack_f[1], stack_conf[1])

                compressed_f, compressed_conf = self._merge_all_c(node1, node2)
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
                #fixme:
                # node1_confs = node1[1][compressed_conf[:, 0]]
                # node2_confs = node2[1][compressed_conf[:, 1]]
                # confs = np.hstack((node1_confs, node2_confs))
                # assert confs.shape[0] == compressed_f.shape[0]
                merged_confs = []
                for i, choices in enumerate(compressed_conf):
                    # each choice is under each theta_c
                    node1_confs = node1[1][i][choices[:, 0]]
                    node2_confs = node2[1][i][choices[:, 1]]
                    confs = np.hstack((node1_confs, node2_confs))
                    assert confs.shape[0] == choices.shape[0]
                    merged_confs.append(confs)

                split_confs = [np.hsplit(x, len(stages_inds)) for x in merged_confs]
                sort_inds = np.argsort(stages_inds)
                # let confs follow the order of qs0, 1, ...
                final_confs = [np.concatenate([y[x] for x in sort_inds], axis=1) for y in split_confs]

                assert len(final_confs) == len(compressed_f)
                assert all([x.shape[0] == y.shape[0] for x, y in zip(compressed_f, final_confs)])
                assert all([x.shape[1] == len(stages_inds) * self.len_theta for x in final_confs])

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
                            stack_f[i * 2],
                            stack_conf[i * 2],
                        )
                        node2 = (
                            stack_f[i * 2 + 1],
                            stack_conf[i * 2 + 1],
                        )
                        compressed_f, compressed_conf = self._merge_all_c(node1, node2)

                        new_stack_f_list.append(compressed_f)

                        stages_str.append(f"{node1_qs_id}+{node2_qs_id}")
                        stages_str.remove(node1_qs_id)
                        stages_str.remove(node2_qs_id)

                        merged_confs = []
                        for i, choices in enumerate(compressed_conf):
                            # each choice is under each theta_c
                            node1_confs = node1[1][i][choices[:, 0]]
                            node2_confs = node2[1][i][choices[:, 1]]
                            confs = np.hstack((node1_confs, node2_confs))
                            assert confs.shape[0] == choices.shape[0]
                            merged_confs.append(confs)

                        new_stack_conf_list.append(merged_confs)

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

    def _compute_pareto_frontier_efficient_all_c(
        self,
        all_objs: List[List[Any]],
        all_confs: List[Tuple[Any, ...]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        po_objs_list = []
        po_confs_list = []
        for objs, confs in zip(all_objs, all_confs):
            objs_arr = np.array(objs)
            po_inds = is_pareto_efficient(objs_arr).tolist()
            po_objs = objs_arr[po_inds]
            po_confs = np.array(confs)[po_inds]

            po_objs_list.append(po_objs)
            po_confs_list.append(po_confs)

        return po_objs_list, po_confs_list

    def _compute_all_configurations(
        self,
        node1: Tuple[np.ndarray, np.ndarray],
        node2: Tuple[np.ndarray, np.ndarray],
    ) -> Tuple[List[List[Any]], List[Tuple[Any, ...]]]:
        # node1_f = np.unique(node1[0], axis=0)
        # node2_f = np.unique(node2[0], axis=0)
        node1_f = node1[0]
        node2_f = node2[0]
        pre_product = [list(range(node1_f.shape[0])), list(range(node2_f.shape[0]))]

        def mass_evaluate(raw_conf: Tuple[Any, ...]) -> List[Any]:
            i, j = raw_conf[0], raw_conf[1]
            latency, cost = np.sum([node1_f[i], node2_f[j]], axis=0, dtype=np.float64)
            return [latency, cost]

        all_confs = list(itertools.product(*pre_product))
        objs = list(map(mass_evaluate, all_confs))
        confs = all_confs

        return objs, confs

    def _compute_all_configurations_all_c(
        self,
        node1: Tuple[np.ndarray, np.ndarray],
        node2: Tuple[np.ndarray, np.ndarray],
    ) -> Tuple[List[List[Any]], List[Tuple[Any, ...]]]:

        all_confs_tuple = [list(itertools.product(*[list(range(x.shape[0])), list(range(y.shape[0]))])) for x, y in
         zip(node1[0], node2[0])]
        # all_confs_list = [list(sum(x, ())) for x in all_confs_tuple]
        # max_length = max([len(x) for x in all_confs_list])
        # pad_all_confs_list = [x + [-1] * (max_length - len(x)) for x in all_confs_list]
        # all_confs_arr = np.array(pad_all_confs_list, dtype=object)
        # assert all_confs_arr.shape[1] == max_length

        def mass_evaluate(raw_conf: Tuple[Any, ...]) -> List[Any]:
            i, j = raw_conf[0], raw_conf[1]
            latency, cost = np.sum([node1_f[i], node2_f[j]], axis=0, dtype=np.float64)
            return [latency, cost]

        objs_list = []
        # node1, node2 include solutions of all theta_c
        # node1 or node2: list, each item is a solution array under one theta_c
        # fixme: whether/how to vectorize the following for-loop
        for i, (node1_f, node2_f) in enumerate(zip(node1[0], node2[0])):
            all_confs = all_confs_tuple[i]
            objs = list(map(mass_evaluate, all_confs))
            objs_list.append(objs)

        confs = all_confs_tuple

        return objs_list, confs

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
