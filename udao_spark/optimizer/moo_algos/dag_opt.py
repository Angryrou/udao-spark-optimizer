# Copyright (c) 2020 École Polytechnique
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Author(s): Qi FAN <qi dot fan at polytechnique dot edu>
#
# Description: DAG optimization in query_moo_general (DAG is degenerated as list)
#
# Created at 21/11/2023
import itertools
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
import pygmo as pg  # type: ignore
from tqdm import tqdm  # type: ignore


class DAGOpt:
    def __init__(
        self,
        f: pd.DataFrame,
        conf: pd.DataFrame,
        algo: str,
        runmode: str,
    ) -> None:
        # key: stage_id, value: dict{c_ind: stage-level Pareto solutions}
        self.f = f
        self.conf = conf
        # hier_moo; seq_div_and_conq; seq_div_and_conq(approx); approx_solve
        self.algo = algo
        self.runmode = runmode

        # self.n_stages = len(f)

    def solve(self) -> Tuple[np.ndarray, pd.DataFrame]:
        if "hier_moo" in self.algo:
            # weights
            n_ws = int(self.algo.split("%")[1])
            n_objs = 2
            ws_steps = 1 / (int(n_ws) - 1)
            ws_pairs = self.even_weights(ws_steps, n_objs)
            F, Theta = self._hier_moo(self.f, self.conf, ws_pairs)
        elif self.algo == "seq_div_and_conq":
            F, Theta = self._seq_div_and_conq(self.f, self.conf, mode="all")
        elif self.algo == "seq_div_and_conq(approx)":
            F, Theta = self._seq_div_and_conq(self.f, self.conf, mode="approx")
        elif self.algo == "approx_solve":
            F, Theta = self.approx_solve(self.f, self.conf)
        else:
            raise Exception(f"mode {self.algo} is not supported in {classmethod}!")

        return F, Theta

    # ------------Approximate solve------------------------------#
    def approx_solve(
        self, f_df: pd.DataFrame, conf_df: pd.DataFrame
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        # find \theta_c with just one optimal \theta_s
        qs_indices = np.array(f_df.index.values.tolist())
        stages = np.unique(qs_indices[:, 0])
        c_inds_one_arr, c_inds_multi_arr = self._get_c_inds(f_df, stages)

        query_f_df_one, query_conf_df_one = self._solve_c_inds_one(
            f_df, conf_df, c_inds_one_arr, stages
        )

        f_multi_list = []
        conf_multi_list = []

        for i in tqdm(c_inds_multi_arr, total=c_inds_multi_arr.shape[0]):
            query_min_ana_lat, query_min_io = np.zeros(
                2,
            ), np.zeros(
                2,
            )
            query_conf_df_list = []
            for stage_id in stages:
                qs_f = f_df.query(f"qs_id == {stage_id}").query(f"c_id == {i}")
                qs_conf = conf_df.query(f"qs_id == {stage_id}").query(f"c_id == {i}")
                min_ind_ana_lat = np.argmin(qs_f.values[:, 0])
                min_ind_io = np.argmin(qs_f.values[:, 1])
                # query_min_ana_lat += qs_f.iloc[min_ind_ana_lat, :]
                # query_min_io += qs_f.iloc[min_ind_io, :]
                query_min_ana_lat += qs_f.values[min_ind_ana_lat, :]
                query_min_io += qs_f.values[min_ind_io, :]
                # query_conf_df_list.append(qs_conf.iloc[min_ind_ana_lat, :])
                # query_conf_df_list.append(qs_conf.iloc[min_ind_io, :])
                query_conf_df_list.append(qs_conf.values[min_ind_ana_lat, :])
                query_conf_df_list.append(qs_conf.values[min_ind_io, :])

            query_min_ana_lat_df = pd.DataFrame(
                query_min_ana_lat, index=np.arange(query_min_ana_lat.shape[0])
            )
            query_min_io_df = pd.DataFrame(
                query_min_io, index=np.arange(query_min_io.shape[0])
            )
            i_query_f_df = pd.concat([query_min_ana_lat_df, query_min_io_df], axis=1).T
            f_multi_list.append(i_query_f_df)

            i_query_conf_values = np.vstack(query_conf_df_list)
            index = [
                stages.repeat(2),
                np.ones(
                    stages.shape[0] * 2,
                )
                * i,
            ]
            indices = pd.MultiIndex.from_tuples(
                list(zip(*index)), names=["qs_id", "c_id"]
            )
            i_query_conf_df_w_index = pd.DataFrame(i_query_conf_values, index=indices)
            conf_multi_list.append(i_query_conf_df_w_index)

        # filter dominated among all theta_c candidates
        all_query_f_df = pd.concat(f_multi_list + query_f_df_one)
        all_conf_df = pd.concat(conf_multi_list + query_conf_df_one)
        assert np.all(
            [
                all_conf_df.query(f"qs_id == {stage_id}").shape[0]
                == all_query_f_df.shape[0]
                for stage_id in stages
            ]
        )

        po_query_ind = pg.non_dominated_front_2d(all_query_f_df)
        po_query_f_df = all_query_f_df.iloc[po_query_ind, :]
        po_conf_df_list = [
            all_conf_df.query(f"qs_id == {stage_id}").iloc[po_query_ind, :]
            for stage_id in stages
        ]

        return po_query_f_df.values, pd.concat(po_conf_df_list)

    # -------------General Hierarchical MOO-----------------------#
    def _hier_moo(
        self,
        f_df: pd.DataFrame,
        conf_df: pd.DataFrame,
        ws_pairs: List[List[Any]],
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        # find \theta_c with just one optimal \theta_s
        qs_indices = np.array(f_df.index.values.tolist())
        stages = np.unique(qs_indices[:, 0])
        c_inds_one_arr, c_inds_multi_arr = self._get_c_inds(f_df, stages)

        query_f_df_one, query_conf_df_one = self._solve_c_inds_one(
            f_df, conf_df, c_inds_one_arr, stages
        )

        f_multi_list = []
        conf_multi_list = []
        for i in tqdm(c_inds_multi_arr, total=c_inds_multi_arr.shape[0]):
            c_i_f_df_multi_list = [
                f_df.query(f"qs_id == {stage_id}").query(f"c_id == {i}")
                for stage_id in stages
            ]
            c_i_conf_df_multi_list = [
                conf_df.query(f"qs_id == {stage_id}").query(f"c_id == {i}")
                for stage_id in stages
            ]

            po_objs, po_confs = self._ws_all_sum_nodes(
                i, c_i_f_df_multi_list, c_i_conf_df_multi_list, ws_pairs, stages
            )
            f_multi_list.append(po_objs)
            conf_multi_list.append(po_confs)

        # filter dominated among all theta_c candidates
        all_query_f_df = pd.concat(f_multi_list + query_f_df_one)
        all_conf_df = pd.concat(conf_multi_list + query_conf_df_one)
        assert np.all(
            [
                all_conf_df.query(f"qs_id == {stage_id}").shape[0]
                == all_query_f_df.shape[0]
                for stage_id in stages
            ]
        )

        po_query_ind = pg.non_dominated_front_2d(all_query_f_df)
        po_query_f_df = all_query_f_df.iloc[po_query_ind, :]
        po_conf_df_list = [
            all_conf_df.query(f"qs_id == {stage_id}").iloc[po_query_ind, :]
            for stage_id in stages
        ]

        return po_query_f_df.values, pd.concat(po_conf_df_list)

    def _ws_all_sum_nodes(
        self,
        i: int,
        f_df_list: List[pd.DataFrame],
        conf_df_list: List[pd.DataFrame],
        ws_pairs: List[List[Any]],
        stages: np.ndarray,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # i is the index of c
        po_obj_list = []
        po_var_list = []

        f_df = pd.concat(f_df_list)
        conf_df = pd.concat(conf_df_list)

        for ws in ws_pairs:
            sum_objs = np.zeros(
                [
                    2,
                ]
            )
            sel_conf_list = []

            for stage_id in stages:
                objs = f_df.query(f"qs_id == {stage_id}").query(f"c_id == {i}").values
                objs_min, objs_max = objs.min(0), objs.max(0)
                if all((objs_min - objs_max) <= 0):
                    obj = np.sum(objs * ws_pairs, axis=1)
                    po_ind = int(np.argmin(obj))
                    # po_ind = self._get_soo_index(objs, ws)
                    sum_objs += objs[po_ind]
                    sel_conf = (
                        conf_df.query(f"qs_id == {stage_id}")
                        .query(f"c_id == {i}")
                        .iloc[po_ind, :]
                    )
                    sel_conf_list.append(sel_conf)
                else:
                    raise Exception(
                        "Cannot do normalization! "
                        "Lower bounds of objective values "
                        "are higher than its upper bounds."
                    )

            po_obj_list.append(sum_objs.tolist())

            po_conf_i = pd.concat(sel_conf_list, axis=1).T
            index_i = [
                stages,
                np.ones(
                    stages.shape[0],
                )
                * i,
            ]
            indices = pd.MultiIndex.from_tuples(
                list(zip(*index_i)), names=["qs_id", "c_id"]
            )
            po_var_list.append(pd.DataFrame(po_conf_i, index=indices))

        # only keep non-dominated solutions
        i_po_ind = pg.non_dominated_front_2d(po_obj_list)
        i_query_objs = np.array(po_obj_list)[i_po_ind]

        i_query_f_df = pd.DataFrame(
            i_query_objs,
            index=np.ones(
                i_query_objs.shape[0],
            )
            * i,
        )
        i_query_conf_df = pd.concat(po_var_list)

        return i_query_f_df, i_query_conf_df

    # def _get_soo_index(
    #     self, objs: np.ndarray, ws_pairs: List[Any]
    # ) -> np.ndarray[Any, Any]:
    #     """
    #     reuse code in VLDB2022
    #     :param objs: ndarray(n_feasible_samples/grids, 2)
    #     :param ws_pairs: list, one weight setting for all objectives, e.g. [0, 1]
    #     :return: int, index of the minimum weighted sum
    #     """
    #     obj = np.sum(objs * ws_pairs, axis=1)
    #     return np.argmin(obj)

    def _get_c_inds(
        self, f_df: pd.DataFrame, stages: np.ndarray
    ) -> Tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        init_c_inds_one_arr = np.array([])
        final_c_inds_one_arr = np.array([])

        indices_arr = np.array(
            f_df.query(f"qs_id == {stages[0]}").index.values.tolist()
        )
        uniq_c_inds = np.unique(indices_arr[:, 1])

        for stage_id in stages:
            inds_qs_arr = np.array(
                f_df.query(f"qs_id == {stage_id}").index.values.tolist()
            )
            c_inds_qs = inds_qs_arr[:, 1]
            # check: all the QS should have the same c_inds
            assert np.all(uniq_c_inds == np.unique(c_inds_qs))

            c_inds_one = np.where(np.unique(c_inds_qs, return_counts=True)[1] == 1)[0]

            if init_c_inds_one_arr.shape[0] == 0:
                final_c_inds_one_arr = c_inds_one
                init_c_inds_one_arr = c_inds_one
            else:
                final_c_inds_one_arr = np.intersect1d(final_c_inds_one_arr, c_inds_one)

        if final_c_inds_one_arr.shape[0] == 0:
            c_inds_multi_arr = uniq_c_inds
        else:
            mask = ~np.isin(uniq_c_inds, final_c_inds_one_arr)
            c_inds_multi_arr = uniq_c_inds[mask]

        return final_c_inds_one_arr, c_inds_multi_arr

    def _solve_c_inds_one(
        self,
        f_df: pd.DataFrame,
        conf_df: pd.DataFrame,
        c_inds_one_list: np.ndarray,
        stages: np.ndarray,
    ) -> Tuple[List[Any], List[Any]]:
        if c_inds_one_list.shape[0] == 0:
            return [], []
        else:
            reshape_qs_objs = np.hstack(
                [
                    f_df.query(f"qs_id == {i}")
                    .reset_index(level=0, drop=True)
                    .loc[c_inds_one_list, :]
                    .values
                    for i in stages
                ]
            )
            query_lat = reshape_qs_objs[:, ::2].sum(1)
            query_io = reshape_qs_objs[:, 1::2].sum(1)
            query_objs = np.vstack((query_lat, query_io)).T

            query_f_df_one = pd.DataFrame(query_objs, index=c_inds_one_list)

            query_conf_list = [
                conf_df.query(f"qs_id == {i}")
                .reset_index(level=0, drop=True)
                .loc[c_inds_one_list, :]
                .values
                for i in stages
            ]

            query_conf_df_one = pd.concat(
                [
                    pd.DataFrame(
                        conf,
                        index=pd.MultiIndex.from_tuples(
                            list(
                                zip(
                                    *[
                                        np.ones(
                                            c_inds_one_list.shape[0],
                                        )
                                        * i,
                                        c_inds_one_list,
                                    ]
                                )
                            ),
                            names=["qs_id", "c_id"],
                        ),
                    )
                    for i, conf in enumerate(query_conf_list)
                ]
            )

            return [query_f_df_one], [query_conf_df_one]

    # -------------Sequential Divide-and-Conquer-------------------#
    def _seq_div_and_conq(
        self,
        f_df: pd.DataFrame,
        conf_df: pd.DataFrame,
        mode: str = "all",
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        indices_arr = np.array(f_df.index.values.tolist())
        stages = np.unique(indices_arr[:, 0])
        c_inds_one_arr, c_inds_multi_arr = self._get_c_inds(f_df, stages)

        query_f_df_one, query_conf_df_one = self._solve_c_inds_one(
            f_df, conf_df, c_inds_one_arr, stages
        )

        f_multi_list = []
        conf_multi_list = []
        for i in tqdm(c_inds_multi_arr, total=c_inds_multi_arr.shape[0]):
            c_i_f_df_multi_list = [
                f_df.query(f"qs_id == {stage_id}").query(f"c_id == {i}")
                for stage_id in stages
            ]
            c_i_conf_df_multi_list = [
                conf_df.query(f"qs_id == {stage_id}").query(f"c_id == {i}")
                for stage_id in stages
            ]
            if mode == "all":
                query_f_df_multi, query_conf_df_multi = self._divide_and_conquer(
                    c_i_f_df_multi_list, c_i_conf_df_multi_list
                )
            elif mode == "approx":
                sample_f_df, sample_conf_df = self._approximate_frontier(
                    c_i_f_df_multi_list, c_i_conf_df_multi_list
                )
                query_f_df_multi, query_conf_df_multi = self._divide_and_conquer(
                    sample_f_df, sample_conf_df
                )
                query_f_df_multi1, query_conf_df_multi1 = self._divide_and_conquer(
                    sample_f_df, sample_conf_df, mode="non_recur"
                )
                assert np.all(query_f_df_multi.values == query_f_df_multi1.values)
            else:
                raise Exception(f"mode {mode} is not supported in Seq_Div_and_Conq!")

            f_multi_list.append(query_f_df_multi)
            conf_multi_list.append(query_conf_df_multi)

        # filter dominated among all \theta_c
        all_query_f_df = pd.concat(f_multi_list + query_f_df_one)
        all_conf_df = pd.concat(conf_multi_list + query_conf_df_one)
        assert np.all(
            [
                all_conf_df.query(f"qs_id == {stage_id}").shape[0]
                == all_query_f_df.shape[0]
                for stage_id in stages
            ]
        )

        po_query_ind = pg.non_dominated_front_2d(all_query_f_df)
        po_query_f_df = all_query_f_df.iloc[po_query_ind, :]
        po_conf_df_list = [
            all_conf_df.query(f"qs_id == {stage_id}").iloc[po_query_ind, :]
            for stage_id in stages
        ]

        return po_query_f_df.values, pd.concat(po_conf_df_list)

    def _merge(
        self,
        node1: Tuple[pd.DataFrame, pd.DataFrame],
        node2: Tuple[pd.DataFrame, pd.DataFrame],
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        all_objs, all_confs = self._compute_all_configurations(node1, node2)
        po_objs, po_confs = self._compute_pareto_frontier_efficient(
            all_objs, all_confs, node1, node2
        )
        return po_objs, po_confs

    def _divide_and_conquer(
        self,
        f: List[pd.DataFrame],
        conf: List[pd.DataFrame],
        mode: str = "recur",
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if mode == "recur":
            return self._divide_and_conquer_recur(f, conf)
        else:
            return self._div_and_conq_non_recur(f, conf)

    def _divide_and_conquer_recur(
        self, f: List[pd.DataFrame], conf: List[pd.DataFrame]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if len(f) == 1:
            result = (f[0], conf[0])
            return result
        if len(f) == 2:
            return self._merge((f[0], conf[0]), (f[1], conf[1]))

        m = len(f) // 2
        left = self._divide_and_conquer_recur(f[:m], conf[:m])
        right = self._divide_and_conquer_recur(f[m:], conf[m:])

        result_f, result_conf = self._merge(left, right)
        return result_f, result_conf

    def _div_and_conq_non_recur(
        self, f: List[pd.DataFrame], conf: List[pd.DataFrame]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        stack_f = f.copy()
        stack_conf = conf.copy()

        while stack_f:
            if len(stack_f) == 2 and len(stack_conf) == 2:
                node1 = (stack_f[0], stack_conf[0])
                node2 = (stack_f[1], stack_conf[1])

                result = self._merge(node1, node2)
                break

            else:
                # get sub_problems
                n_sub_problems = (
                    int(len(stack_f) / 2)
                    if len(stack_f) % 2 == 0
                    else int(len(stack_f) / 2) + 1
                )

                new_stack_f_df_list = []
                new_stack_conf_df_list = []
                for i in range(n_sub_problems):
                    if i * 2 + 1 == len(stack_f):
                        new_stack_f_df_list.append(stack_f[-1])
                        new_stack_conf_df_list.append(stack_conf[-1])
                    else:
                        node1 = (stack_f[i * 2], stack_conf[i * 2])
                        node2 = (stack_f[i * 2 + 1], stack_conf[i * 2 + 1])
                        compressed_f_df, compressed_conf_df = self._merge(node1, node2)
                        new_stack_f_df_list.append(compressed_f_df)
                        new_stack_conf_df_list.append(compressed_conf_df)

                stack_f = new_stack_f_df_list
                stack_conf = new_stack_conf_df_list

        return result

    def _compute_pareto_frontier_efficient(
        self,
        all_objs: List[List[Any]],
        all_confs: List[Tuple[Any, ...]],
        node1: Tuple[pd.DataFrame, pd.DataFrame],
        node2: Tuple[pd.DataFrame, pd.DataFrame],
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        all_indices = np.array(node1[1].index.values.tolist())
        c_ind = np.unique(all_indices[:, 1])

        arr_points = np.array(all_objs)
        # indices = moo_ut.is_pareto_efficient(arr_points)
        indices = pg.non_dominated_front_2d(arr_points).tolist()
        po_objs = arr_points[indices]
        po_f_df = pd.DataFrame(
            po_objs,
            index=np.ones(
                po_objs.shape[0],
            )
            * c_ind,
        )

        # all_confs: column1: selections in node1, column2: selections in node2
        po_confs = np.array(all_confs)[indices]

        qs_all_indices_node1 = np.array(node1[1].index.values.tolist())
        qs_all_indices_node2 = np.array(node2[1].index.values.tolist())
        stages_node1 = np.unique(qs_all_indices_node1[:, 0])
        stages_node2 = np.unique(qs_all_indices_node2[:, 0])

        # all_confs: column1: selections in node1, column2: selections in node2
        selections_node1 = po_confs[:, 0]
        selections_node2 = po_confs[:, 1]

        po_conf_node1 = [
            node1[1]
            .query(f"qs_id == {stage_id}")
            .query(f"c_id == {c_ind}")
            .iloc[sel, :]
            for stage_id in stages_node1
            for sel in selections_node1
        ]
        po_conf_node2 = [
            node2[1]
            .query(f"qs_id == {stage_id}")
            .query(f"c_id == {c_ind}")
            .iloc[sel, :]
            for stage_id in stages_node2
            for sel in selections_node2
        ]

        index_node1 = [
            stages_node1.repeat(selections_node1.shape[0]),
            np.array(
                np.ones(
                    stages_node1.shape[0],
                )
                * c_ind
            ).repeat(selections_node1.shape[0]),
        ]
        indices_node1 = pd.MultiIndex.from_tuples(
            list(zip(*index_node1)), names=["qs_id", "c_id"]
        )
        po_conf_node1_df = pd.DataFrame(po_conf_node1, index=indices_node1)

        index_node2 = [
            stages_node2.repeat(selections_node2.shape[0]),
            np.array(
                np.ones(
                    stages_node2.shape[0],
                )
                * c_ind
            ).repeat(selections_node2.shape[0]),
        ]
        indices_node2 = pd.MultiIndex.from_tuples(
            list(zip(*index_node2)), names=["qs_id", "c_id"]
        )
        po_conf_node2_df = pd.DataFrame(po_conf_node2, index=indices_node2)

        return po_f_df, pd.concat([po_conf_node1_df, po_conf_node2_df])

    def _compute_all_configurations(
        self,
        node1: Tuple[pd.DataFrame, pd.DataFrame],
        node2: Tuple[pd.DataFrame, pd.DataFrame],
    ) -> Tuple[List[List[Any]], List[Tuple[Any, ...]]]:
        pre_product = [list(range(len(node1[0]))), list(range(len(node2[0])))]

        def mass_evaluate(raw_conf: Tuple[Any, ...]) -> List[Any]:
            i, j = raw_conf[0], raw_conf[1]
            latency, cost = np.sum(
                [node1[0].values[i], node2[0].values[j]], axis=0, dtype=np.float64
            )
            return [latency, cost]

        all_confs = list(itertools.product(*pre_product))
        objs = list(map(mass_evaluate, all_confs))
        confs = all_confs

        return objs, confs

    def _approximate_frontier(
        self,
        f_df_list: List[pd.DataFrame],
        conf_df_list: List[pd.DataFrame],
    ) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
        product_size = np.prod([v.shape[0] for v in f_df_list])

        target_size = 3_000

        approx_f_df = f_df_list
        approx_conf_df = conf_df_list

        while product_size > target_size:
            # find the QS with the largest size
            top_stage_id = np.argmax([[v.shape[0] for v in approx_f_df]])
            # randomly sample 1/2 of the frontier of the above QS
            top_size = approx_f_df[top_stage_id].shape[0]
            random_choices = np.random.choice(
                range(top_size), top_size // 2, replace=False
            )
            # update product_size, approx_f and approx_conf
            approx_f_df[top_stage_id] = approx_f_df[top_stage_id].iloc[
                random_choices, :
            ]
            approx_conf_df[top_stage_id] = approx_conf_df[top_stage_id].iloc[
                random_choices, :
            ]
            product_size = np.prod([v.shape[0] for v in approx_f_df])

        return approx_f_df, approx_conf_df

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
