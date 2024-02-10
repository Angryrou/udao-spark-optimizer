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
# Created at 04/01/2024

import random
import signal
import time
from dataclasses import dataclass
from multiprocessing import Pool

# from torch.multiprocessing import Pool
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# import numba as nb
import numpy as np
import pandas as pd
import pygmo as pg  # type: ignore
import torch as th

# th.multiprocessing.set_sharing_strategy('file_system')
# from numba.typed import List
from sklearn.cluster import KMeans
from sklearnex import patch_sklearn  # type: ignore

from udao_spark.optimizer.moo_algos.dag_opt import DAGOpt

patch_sklearn()


class DivAndConqMOO:
    @dataclass
    class Params:
        c_samples: Union[np.ndarray, th.Tensor]
        "samples for theta_c"
        p_samples: Union[np.ndarray, th.Tensor]
        "samples for theta_p"
        s_samples: Union[np.ndarray, th.Tensor]
        "samples for theta_s"
        n_clusters: int
        "the number of clusters"
        cross_location: int
        "the crossover location"
        dag_opt_algo: str
        "the algrorithm of dag optimization"
        tune_p_solver: str = "grid"
        "the solver of tuning theta_p under each fixed theta_c within a stage"
        cluster_algo: str = "k-means"
        "the clustering algorithm"
        flag_filter_c: bool = False
        "the flag to indicate whether to filter theta_c in before dag_opt"
        runmode: str = "multiprocessing"
        "to indicate whether to add multiprocessing or to solve in a naive way"
        time_limit: int = -1
        "the time limit of running compile-time optimization (-1: no-limit)"
        verbose: bool = False
        "flag to indicate whether to print more info"

    def __init__(
        self,
        n_stages: int,
        graph_embeddings: th.Tensor,
        non_decision_tabular_features: Union[pd.DataFrame, th.Tensor],
        # non_decision_tabular_features: Any,
        obj_model: Callable[[Any, Any, Any, Any], Any],
        params: Params,
        use_ag: bool,
        ag_model: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.n_stages = n_stages
        self.graph_embeddings = graph_embeddings
        self.non_decision_tabular_features = non_decision_tabular_features
        self.seed = seed
        self.device = th.device("cuda") if th.cuda.is_available() else th.device("cpu")

        self.c_samples = params.c_samples
        self.p_samples = params.p_samples
        self.s_samples = params.s_samples
        self.cluster_algo = params.cluster_algo
        self.n_clusters = params.n_clusters
        self.cross_location = params.cross_location
        self.obj_model = obj_model
        self.flag_filter_c = params.flag_filter_c
        self.dag_opt_algo = params.dag_opt_algo
        self.runmode = params.runmode
        self.time_limit = params.time_limit
        self.verbose = params.verbose

        self.ag_model = ag_model
        self.use_ag = use_ag

        self.len_theta_c = params.c_samples.shape[1]
        self.len_theta = (
            params.c_samples.shape[1]
            + params.p_samples.shape[1]
            + params.s_samples.shape[1]
        )
        self.n_objs = 2

    def solve(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.time_limit > 0:

            def signal_handler(signum: Any, frame: Any) -> None:
                raise Exception("Timed out!")

            signal.signal(signal.SIGALRM, signal_handler)
            signal.alarm(self.time_limit)

            try:
                F, Theta = self.div_moo_solve()
                signal.alarm(
                    0
                )  ##cancel the timer if the function returned before timeout

            except Exception:
                F = -1 * np.ones((1, 2))
                Theta = np.array([[-1], [-1]])
                print("Timed out for query")
        else:
            F, Theta = self.div_moo_solve()

        return F, Theta

    def div_moo_solve(self) -> Tuple[np.ndarray, np.ndarray]:
        start_qs_tuning = time.time()
        f_th, conf_th, indices_arr = self.get_qs_set()
        if self.verbose:
            print(
                f"time cost of getting effective set is {time.time() - start_qs_tuning}"
            )

        start_dag_opt = time.time()
        F, Theta = self.dag_opt(f_th, conf_th, indices_arr)
        if self.verbose:
            print(f"time cost of dag optimization is {time.time() - start_dag_opt}")
        return F, Theta

    def get_qs_set(self) -> Tuple[th.Tensor, th.Tensor, np.ndarray]:
        # 1. randomly sample each QS
        clustering_features = self.c_samples
        p_samples, s_samples = self.p_samples, self.s_samples

        # 2. cluster_based optimal theta_p and theta_p_s estimation
        (
            tuned_f_th,
            tuned_conf_th,
            indices_arr,
            cluster_model,
            label_rep_theta_c_mapping,
        ) = self._cluster_based_estimate_opt_theta_p_s(
            self.n_clusters,
            clustering_features,
            p_samples,
            s_samples,
            self.cluster_algo,
        )

        # 3. theoretical result 2: get local optimal theta_c
        start_union_c = time.time()
        # union_opt_theta_c_list = self._union_opt_theta_c(tuned_f_df, tuned_conf_df)
        union_opt_theta_c_list = self._union_opt_theta_c(
            tuned_f_th, tuned_conf_th, indices_arr
        )
        if self.verbose:
            print(
                f"FUNCTION: time cost of union theta_c "
                f"is: {time.time() - start_union_c}"
            )
            print()

        # 4. corssover enrichment: use crossover to enrich new \theta_c
        start_extend_c = time.time()
        new_theta_c_list = self._extend_c(
            self.cross_location, union_opt_theta_c_list.copy()
        )
        if self.verbose:
            print(
                f"the number of new generated theta_c candidates "
                f"is {len(new_theta_c_list)}"
            )
            print(
                f"FUNCTION: time cost of extending c is {time.time() - start_extend_c}"
            )
            print()

        # 5. cluster-based theta_p_s estimation
        start_predict = time.time()
        # predict labels
        if len(new_theta_c_list) == 0:
            new_theta_c_list = union_opt_theta_c_list

        pred_cluster_label = cluster_model.predict(new_theta_c_list)  # type: ignore
        if self.verbose:
            print(f"time cost of predicting labels is: {time.time() - start_predict}")

        uniq_cluster_label = np.unique(pred_cluster_label)
        # colume 1: cluster id; colume 2: new theta_c id
        new_label_arr = np.vstack(
            (
                pred_cluster_label,
                np.arange(pred_cluster_label.shape[0]) + clustering_features.shape[0],
            )
        ).T
        new_cluster_members = [
            new_label_arr[:, 1][np.where(new_label_arr[:, 0] == label)].tolist()
            for label in uniq_cluster_label
        ]
        if self.verbose:
            print(
                f"FUNCTION: time cost of predicting cluster labels for new theta_c "
                f"is {time.time() - start_predict}"
            )
            print()

        start_est_opt_p_new_c = time.time()
        new_label_rep_theta_c_mapping = {
            k: label_rep_theta_c_mapping[k] for k in uniq_cluster_label
        }
        (
            new_tuned_f_th,
            new_tuned_conf_th,
            new_indices_arr,
        ) = self._estimate_opt_theta_p_s(
            th.Tensor(new_theta_c_list),
            new_cluster_members,
            new_label_rep_theta_c_mapping,
            tuned_conf_th,
            indices_arr,
            mode="estimate_new",
        )

        if self.verbose:
            print(
                f"FUNCTION: time cost of estimating optimal theta_p for new theta_c "
                f"is {time.time() - start_est_opt_p_new_c}"
            )
            print()
        # 6. union all solutions
        start_union_all = time.time()
        if isinstance(tuned_f_th, np.ndarray):
            all_f_th = np.concatenate([tuned_f_th, new_tuned_f_th])
            all_conf_th = np.concatenate([tuned_conf_th, new_tuned_conf_th])
        else:
            assert isinstance(tuned_f_th, th.Tensor)
            all_f_th = th.concatenate([tuned_f_th, new_tuned_f_th])
            all_conf_th = th.concatenate([tuned_conf_th, new_tuned_conf_th])
        all_indices_arr = np.concatenate([indices_arr, new_indices_arr])

        if self.verbose:
            print(
                f"FUNCTION: time cost of union all "
                f"is {time.time() - start_union_all}"
            )
            print()

        return all_f_th, all_conf_th, all_indices_arr

    def dag_opt(
        self, f_th: th.Tensor, conf_th: th.Tensor, indices_arr: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        dag_opt = DAGOpt(f_th, conf_th, indices_arr, self.dag_opt_algo, self.runmode)

        F, Theta = dag_opt.solve()
        return F, Theta

    # -------get QS effective set------
    def _cluster_based_estimate_opt_theta_p_s(
        self,
        n_clusters: int,
        clustering_features: Union[th.Tensor, np.ndarray],
        p_samples: Union[th.Tensor, np.ndarray],
        s_samples: Union[th.Tensor, np.ndarray],
        cluster_algo: str,
    ) -> Tuple[th.Tensor, th.Tensor, np.ndarray, object, Dict[Any, Any]]:
        # clustering theta_c
        start_cluster = time.time()
        clusters_labels, cluster_model = self._clustering_method(
            n_clusters, clustering_features, theta_c_cluster_algo=cluster_algo
        )
        (
            represent_theta_c,
            cluster_members,
            label_rep_theta_c_mapping,
        ) = self.clustering_representation(clusters_labels)
        if self.verbose:
            print(f"FUNCTION: time cost of cluster is :{time.time() - start_cluster}")
            print()

        start_tune_p = time.time()
        f_th, conf_th, indices_arr = self._tune_p(
            represent_theta_c,
            label_rep_theta_c_mapping,
            clustering_features,
            p_samples,
            s_samples,
            self.n_stages,
        )
        if self.verbose:
            print(
                f"FUNCTION: time cost of tuning theta_p is {time.time() - start_tune_p}"
            )
            print()
        # estimate optimal theta_p and theta_s
        start_est_opt_p = time.time()
        tuned_f_th, tuned_conf_th, tuned_indices_arr = self._estimate_opt_theta_p_s(
            clustering_features,
            cluster_members,
            label_rep_theta_c_mapping,
            conf_th,
            indices_arr,
            mode="estimate_exist",
        )
        if self.verbose:
            print(
                f"FUNCTION: time cost of estimating optimal theta_p for all theta_c "
                f"is {time.time() - start_est_opt_p}"
            )
            print()
        return (
            tuned_f_th,
            tuned_conf_th,
            tuned_indices_arr,
            cluster_model,
            label_rep_theta_c_mapping,
        )

    def _clustering_method(
        self,
        n_cluster: int,
        cluster_features: Union[th.Tensor, np.ndarray],
        theta_c_cluster_algo: str,
    ) -> Tuple[Any, object]:
        if theta_c_cluster_algo == "k-means":
            k_means = KMeans(
                n_clusters=n_cluster, random_state=0, n_init="auto", algorithm="elkan"
            ).fit(cluster_features)
            # print("k-means successful")
            clustering_labels = k_means.labels_
            # print("k-means labels successful")
            return clustering_labels, k_means
        else:
            raise Exception(
                f"Clustering method {theta_c_cluster_algo} is not supported!"
            )

    def clustering_representation(
        self, cluster_labels: np.ndarray
    ) -> Tuple[List[Any], List[List[Any]], Dict[Any, Any]]:
        n_clusters = np.unique(np.array(cluster_labels))
        n_clusters_init = cluster_labels

        new_clusters = []
        new_cluster_members = []

        label_rep_theta_c_mapping = dict()

        for index, item in enumerate(n_clusters):
            indexes = [j for j, x in enumerate(n_clusters_init) if x == item]
            inst_cluster_pick_representative = "random"
            if inst_cluster_pick_representative == "random":
                # randomly choose one instance to represent the current cluster
                random.seed(len(cluster_labels))
                random_choice = random.choice(indexes)
                new_inst_index = [random_choice]
                new_cluster_members.append(indexes)

                new_clusters.extend(new_inst_index)
                label_rep_theta_c_mapping[item] = new_inst_index[0]
            else:
                raise NotImplementedError(inst_cluster_pick_representative)

        return new_clusters, new_cluster_members, label_rep_theta_c_mapping

    def _estimate_opt_theta_p_s(
        self,
        clustering_features: Union[th.Tensor, np.ndarray],
        cluster_members: List[List[int]],
        label_rep_theta_c_mapping: Dict[Any, Any],
        conf_df: Union[th.Tensor, np.ndarray],
        qs_indices: np.ndarray,
        mode: str = "estimate_exist",
    ) -> Tuple[th.Tensor, th.Tensor, np.ndarray]:
        start_df_query = time.time()
        c_inds_list = list(label_rep_theta_c_mapping.values())
        # col0: qs_id, col1: c_id, col2: p_id
        qs_c_inds_arr = qs_indices[:, :2]
        if isinstance(clustering_features, np.ndarray):
            clustering_features = th.tensor(clustering_features, dtype=th.float32)

        if isinstance(conf_df, np.ndarray):
            conf_df = th.tensor(conf_df, dtype=th.float32)

        opt_p_list = [
            [
                conf_df[np.where((qs_c_inds_arr == [qs_id, c_id]).all(1))].reshape(
                    -1, self.len_theta
                )
                for c_id in c_inds_list
            ]
            for qs_id in range(self.n_stages)
        ]

        if self.verbose:
            print(
                f"time cost of df query in estimate opt_p "
                f"is {time.time() - start_df_query}"
            )
        start_mesh_theta = time.time()
        len_opt_p_per_qs = [[x.shape[0] for x in opt_p] for opt_p in opt_p_list]

        mesh_c_ids_w_opt_p = [
            [
                th.repeat_interleave(th.Tensor(cm), opt_p_len)
                for cm, opt_p_len in zip(cluster_members, x)
            ]
            for x in len_opt_p_per_qs
        ]
        mesh_c_ids_w_qs = [th.cat(x) for x in mesh_c_ids_w_opt_p]
        mesh_qs_ids = [
            th.Tensor([stage_id]).repeat(x.shape[0])
            for stage_id, x in zip(range(self.n_stages), mesh_c_ids_w_qs)
        ]

        if mode == "estimate_exist":
            mesh_theta_c_ori = [
                [
                    th.repeat_interleave(clustering_features[cm], opt_p_len, dim=0)
                    for cm, opt_p_len in zip(cluster_members, x)
                ]
                for x in len_opt_p_per_qs
            ]
        else:
            mesh_theta_c_ori = [
                [
                    th.repeat_interleave(
                        clustering_features[
                            (np.array(cm) - self.c_samples.shape[0]).tolist()
                        ],
                        opt_p_len,
                        dim=0,
                    )
                    for cm, opt_p_len in zip(cluster_members, x)
                ]
                for x in len_opt_p_per_qs
            ]
        mesh_theta_c_all_qs = th.cat([th.cat(x) for x in mesh_theta_c_ori])

        if isinstance(opt_p_list[0][0], np.ndarray):
            mesh_theta_p_s_ori = [
                [
                    th.repeat_interleave(
                        th.Tensor(opt_p)[:, clustering_features.shape[1] :],
                        len(cm),
                        dim=0,
                    )
                    for cm, opt_p in zip(cluster_members, opt_p_qs)
                ]
                for opt_p_qs in opt_p_list
            ]
        else:
            mesh_theta_p_s_ori = [
                [
                    # th.repeat_interleave(
                    #     th.Tensor(opt_p)[:, clustering_features.shape[1] :],
                    #     len(cm),
                    #     dim=0,
                    # )
                    th.Tensor(opt_p)[:, clustering_features.shape[1] :].repeat(
                        len(cm), 1
                    )
                    for cm, opt_p in zip(cluster_members, opt_p_qs)
                ]
                for opt_p_qs in opt_p_list
            ]
        mesh_theta_p_s_all_qs = th.cat([th.cat(x) for x in mesh_theta_p_s_ori])
        assert mesh_theta_c_all_qs.shape[0] == mesh_theta_p_s_all_qs.shape[0]
        mesh_theta = th.cat([mesh_theta_c_all_qs, mesh_theta_p_s_all_qs], dim=1)
        mesh_graph_embedding = th.cat(
            [
                self.graph_embeddings[stage_id]
                .reshape(1, -1)
                .repeat_interleave(x.shape[0], dim=0)
                for stage_id, x in zip(range(self.n_stages), mesh_qs_ids)
            ]
        )
        # non_decision_features = self.non_decision_tabular_features
        mesh_non_decision_tabular_features: Union[th.Tensor, pd.DataFrame]
        if self.use_ag and isinstance(self.ag_model, str):
            assert isinstance(self.non_decision_tabular_features, pd.DataFrame)
            mesh_theta = mesh_theta.numpy()

            mesh_non_decision_tabular_features = pd.concat(
                [
                    self.non_decision_tabular_features.loc[
                        np.repeat(
                            self.non_decision_tabular_features.index[stage_id],  # type: ignore
                            x.shape[0],
                        ).astype(int)
                    ].reset_index(drop=True)
                    for stage_id, x in zip(range(self.n_stages), mesh_qs_ids)
                ]
            )
            assert (
                mesh_non_decision_tabular_features.shape[0]
                == mesh_graph_embedding.shape[0]
            )
        else:
            assert isinstance(self.non_decision_tabular_features, th.Tensor)
            mesh_non_decision_tabular_features = th.cat(
                [
                    self.non_decision_tabular_features[stage_id]
                    .reshape(1, -1)
                    .repeat_interleave(x.shape[0], dim=0)
                    for stage_id, x in zip(range(self.n_stages), mesh_qs_ids)
                ]
            )
        if self.verbose:
            print(f"time cost of mesh theta is {time.time() - start_mesh_theta}")
            print(
                f"FUNCTION: time cost of assign opt_theta_p "
                f"is {time.time() - start_df_query}"
            )

        start_predict = time.time()
        objs = self._get_obj_values(
            mesh_theta.shape[0],
            mesh_graph_embedding,
            mesh_non_decision_tabular_features,
            mesh_theta,
        )

        if self.verbose:
            print(f"time cost of predict in est_opt_p is {time.time() - start_predict}")

        start_set_ind_df = time.time()
        mesh_p_ids = th.cat(
            [
                th.cat(
                    [
                        th.arange(opt_p_len).repeat(len(cm))
                        for cm, opt_p_len in zip(cluster_members, x)
                    ]
                )
                for x in len_opt_p_per_qs
            ]
        ).numpy()
        if self.verbose:
            print(f"time cost of concat p_ids is {time.time() - start_set_ind_df}")
        indices_arr = np.vstack(
            [th.cat(mesh_qs_ids).numpy(), th.cat(mesh_c_ids_w_qs).numpy(), mesh_p_ids]
        ).T
        if self.verbose:
            print(f"time cost of vstack indices is {time.time() - start_set_ind_df}")
            print(
                f"FUNCTION: time cost of compute in est_opt_p "
                f"is {time.time() - start_predict}"
            )
            print()

        return objs, mesh_theta, indices_arr

    def _union_opt_theta_c(
        self,
        tuned_f_th: th.Tensor,
        tuned_conf_th: th.Tensor,
        indices_arr: np.ndarray,
    ) -> List[List[Any]]:
        local_opt_theta_c_list = []
        for stage_id in range(self.n_stages):
            qs_values_inds = np.where(indices_arr[:, 0] == stage_id)[0].tolist()
            qs_values = tuned_f_th[qs_values_inds]
            po_ind = pg.non_dominated_front_2d(qs_values).astype(int)
            po_theta_c = tuned_conf_th[po_ind, : self.c_samples.shape[1]].tolist()
            local_opt_theta_c_list.append(po_theta_c)

        return sum(local_opt_theta_c_list, [])

    def _extend_c(
        self,
        location: int,
        opt_c_list: List[List[Any]],
    ) -> List[List[Any]]:
        # crossover to generate new \theta_c
        uniq_res_c = np.unique(np.array(opt_c_list)[:, :location], axis=0)
        uniq_non_res_c = np.unique(np.array(opt_c_list)[:, location:8], axis=0)

        if uniq_res_c.shape[0] <= uniq_non_res_c.shape[0]:
            theta_res_mesh = np.repeat(uniq_res_c, uniq_non_res_c.shape[0], axis=0)
            theta_c_non_res_mesh = np.tile(uniq_non_res_c, (uniq_res_c.shape[0], 1))
            extend_theta_c = np.hstack((theta_res_mesh, theta_c_non_res_mesh))
        else:
            theta_res_mesh = np.tile(uniq_res_c, (uniq_non_res_c.shape[0], 1))
            theta_c_non_res_mesh = np.repeat(
                uniq_non_res_c, uniq_res_c.shape[0], axis=0
            )
            extend_theta_c = np.hstack((theta_res_mesh, theta_c_non_res_mesh))

        # filter theta_c from the newly generated theta_c candidates,
        # which are already included in the union optimal theta_c candidates
        uniq_new_c = np.unique(extend_theta_c, axis=0)
        matching_rows_opt_c = np.all(
            uniq_new_c[:, None, :] == np.array(opt_c_list)[None, :, :], axis=-1
        )
        matching_indices_opt_c = np.where(matching_rows_opt_c.any(axis=-1))[0]
        all_new_c_inds = np.arange(uniq_new_c.shape[0])
        mask_opt_c = ~np.isin(all_new_c_inds, matching_indices_opt_c)
        filtered_inds_opt_c = all_new_c_inds[mask_opt_c]

        extend_c_filter_dup_opt_c = extend_theta_c[filtered_inds_opt_c]

        # filter theta_c from the newly generated theta_c candidates,
        # which are already included in the initial theta_c candidates,
        # i.e. clustering features
        if isinstance(self.c_samples, np.ndarray):
            matching_rows_init_c = np.all(
                extend_c_filter_dup_opt_c[:, None, :] == self.c_samples[None, :, :],
                axis=-1,
            )
        else:
            assert isinstance(self.c_samples, th.Tensor)
            matching_rows_init_c = np.all(
                extend_c_filter_dup_opt_c[:, None, :]
                == self.c_samples.numpy()[None, :, :],
                axis=-1,
            )
        matching_indices_init_c = np.where(matching_rows_init_c.any(axis=-1))[0]
        all_new_c_inds_init_c = np.arange(extend_c_filter_dup_opt_c.shape[0])
        mask_init_c = ~np.isin(all_new_c_inds_init_c, matching_indices_init_c)
        filtered_inds_init_c = all_new_c_inds_init_c[mask_init_c]

        extend_c_list = extend_c_filter_dup_opt_c[filtered_inds_init_c].tolist()
        return extend_c_list

    def _tune_p(
        self,
        represent_theta_c: List[int],
        label_rep_theta_c_mapping: Dict[Any, Any],
        clustering_features: Union[th.Tensor, np.ndarray],
        p_samples: Union[th.Tensor, np.ndarray],
        s_samples: Union[th.Tensor, np.ndarray],
        n_stages: int,
        filter_mode: str = "naive",
    ) -> Tuple[np.ndarray, Union[th.Tensor, np.ndarray], np.ndarray]:
        rep_c_samples = clustering_features[represent_theta_c]

        start_mesh_theta = time.time()
        n_evals = rep_c_samples.shape[0] * p_samples.shape[0] * n_stages

        mesh_c: Union[np.ndarray, th.Tensor]
        mesh_p: Union[np.ndarray, th.Tensor]
        mesh_s: Union[np.ndarray, th.Tensor]

        if isinstance(rep_c_samples, np.ndarray):
            assert isinstance(p_samples, np.ndarray)
            assert isinstance(s_samples, np.ndarray)
            mesh_c_per_stage = np.repeat(rep_c_samples, p_samples.shape[0], axis=0)
            mesh_c = np.tile(mesh_c_per_stage, (n_stages, 1))

            mesh_p_per_stage = np.tile(p_samples, (rep_c_samples.shape[0], 1))
            mesh_p = np.tile(mesh_p_per_stage, (n_stages, 1))

            mesh_s = np.repeat(s_samples, n_evals, axis=0)
            assert (
                mesh_c.shape[0] == mesh_p.shape[0]
                and mesh_s.shape[0] == mesh_p.shape[0]
            )
            mesh_theta = np.concatenate([mesh_c, mesh_p, mesh_s], axis=1)
        else:
            assert isinstance(p_samples, th.Tensor)
            assert isinstance(s_samples, th.Tensor)
            assert isinstance(rep_c_samples, th.Tensor)
            mesh_c = rep_c_samples.repeat_interleave(p_samples.shape[0], dim=0).repeat(
                n_stages, 1
            )
            mesh_p = p_samples.repeat(rep_c_samples.shape[0], 1).repeat(n_stages, 1)
            mesh_s = s_samples.repeat(n_evals, 1)
            assert (
                mesh_c.shape[0] == mesh_p.shape[0]
                and mesh_s.shape[0] == mesh_p.shape[0]
            )
            mesh_theta = th.cat([mesh_c, mesh_p, mesh_s], dim=1)

        mesh_graph_embeddings = self.graph_embeddings.repeat_interleave(
            p_samples.shape[0] * rep_c_samples.shape[0], dim=0
        )

        mesh_non_decision_tabular_features: Union[th.Tensor, pd.DataFrame]
        if self.use_ag and isinstance(self.ag_model, str):
            assert isinstance(self.non_decision_tabular_features, pd.DataFrame)

            mesh_non_decision_tabular_features = self.non_decision_tabular_features.loc[
                np.repeat(
                    self.non_decision_tabular_features.index,
                    p_samples.shape[0] * rep_c_samples.shape[0],
                )
            ].reset_index(drop=True)
        else:
            assert isinstance(self.non_decision_tabular_features, th.Tensor)
            mesh_non_decision_tabular_features = (
                self.non_decision_tabular_features.repeat_interleave(
                    p_samples.shape[0] * rep_c_samples.shape[0], dim=0
                )
            )
        if self.verbose:
            print(f"time cost of mesh theta is: {time.time() - start_mesh_theta}")

        start_predict = time.time()
        y_hat = self._get_obj_values(
            n_evals,
            mesh_graph_embeddings,
            mesh_non_decision_tabular_features,
            mesh_theta,
        )
        if self.verbose:
            print(
                f"time cost of predict objective values of mesh theta "
                f"is {time.time() - start_predict}"
            )

        start_filter = time.time()
        if isinstance(y_hat, np.ndarray):
            split_times = n_stages * rep_c_samples.shape[0]
            split_y_hat = np.split(y_hat, split_times)
            split_theta = np.split(mesh_theta, split_times)
        else:
            assert isinstance(y_hat, th.Tensor)
            split_y_hat = th.split(y_hat, p_samples.shape[0])
            split_theta = th.split(mesh_theta, p_samples.shape[0])

        conf_list: List[Any] = []

        if filter_mode == "naive":
            f_list = []
            # conf_list = []
            indices_list = []
            for i, (sub_f, sub_conf) in enumerate(zip(split_y_hat, split_theta)):
                f, conf, indices = self._keep_opt_p_per_c(
                    i, rep_c_samples, label_rep_theta_c_mapping, sub_f, sub_conf
                )

                f_list.append(f)
                conf_list.append(conf)
                indices_list.append(indices)
        else:  # not work for multiprocessing with tensor
            arg_list = [
                (i, rep_c_samples, label_rep_theta_c_mapping, sub_f, sub_conf)
                for i, (sub_f, sub_conf) in enumerate(zip(split_y_hat, split_theta))
            ]

            with Pool(processes=20) as pool:
                ret_list = pool.starmap(self._keep_opt_p_per_c, arg_list.copy())

            f_list = [result[0] for result in ret_list]
            conf_list = [result[1] for result in ret_list]
            indices_list = [result[2] for result in ret_list]

        if self.verbose:
            print(
                f"time cost of filtering dominated theta_p of "
                f"all theta_c is {time.time() - start_filter}"
            )

        if self.use_ag and isinstance(self.ag_model, str):
            assert isinstance(f_list[0], np.ndarray)
            assert isinstance(conf_list[0], np.ndarray)
            all_confs_th = np.concatenate(conf_list)
        else:
            assert isinstance(conf_list[0], th.Tensor)
            all_confs_th = th.concatenate(conf_list)
        all_f_th = np.concatenate(f_list)
        all_indices_arr = np.concatenate(indices_list)
        return all_f_th, all_confs_th, all_indices_arr

    def _keep_opt_p_per_c(
        self,
        i: int,
        rep_c_samples: Union[th.Tensor, np.ndarray],
        label_rep_theta_c_mapping: dict,
        sub_f: Union[th.Tensor, np.ndarray],
        sub_conf: Union[th.Tensor, np.ndarray],
    ) -> Tuple[Union[th.Tensor, np.ndarray], Union[th.Tensor, np.ndarray], np.ndarray]:
        stage_id = int(i / rep_c_samples.shape[0])
        c_id = label_rep_theta_c_mapping[int(i % rep_c_samples.shape[0])]
        po_ind = pg.non_dominated_front_2d(sub_f).tolist()
        po_objs = sub_f[po_ind]
        po_confs = sub_conf[po_ind]

        time.time()
        qs_ids = (
            np.ones(
                [
                    po_objs.shape[0],
                ]
            )
            * stage_id
        )
        c_ids = (
            np.ones(
                [
                    po_objs.shape[0],
                ]
            )
            * c_id
        )
        p_ids = np.arange(po_objs.shape[0])
        indices_arr = np.vstack((qs_ids, c_ids, p_ids)).T
        return po_objs, po_confs, indices_arr

    def _get_obj_values(
        self,
        n_evals: int,
        mesh_graph_embeddings: th.Tensor,
        mesh_non_decision_tabular_features: Union[th.Tensor, pd.DataFrame],
        mesh_theta: th.Tensor,
    ) -> th.Tensor:
        print(f"n_evals is {n_evals}")
        if self.use_ag and isinstance(self.ag_model, str):
            y_hat = self.obj_model(
                mesh_graph_embeddings.cpu().numpy(),
                mesh_non_decision_tabular_features,
                mesh_theta,
                self.ag_model,
            )
        else:
            y_hat = self.obj_model(
                mesh_graph_embeddings,
                mesh_non_decision_tabular_features,
                mesh_theta,
                None,
            )
        return y_hat
