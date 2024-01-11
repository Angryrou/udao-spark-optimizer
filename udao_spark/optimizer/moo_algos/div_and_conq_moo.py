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
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pygmo as pg  # type: ignore
import torch as th
from sklearn.cluster import KMeans
from sklearnex import patch_sklearn  # type: ignore
from torch.utils.data import DataLoader, TensorDataset

from udao_spark.optimizer.moo_algos.dag_opt import DAGOpt

patch_sklearn()


class DivAndConqMOO:
    @dataclass
    class Params:
        c_samples: th.Tensor
        "samples for theta_c"
        p_samples: th.Tensor
        "samples for theta_p"
        s_samples: th.Tensor
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

    def __init__(
        self,
        n_stages: int,
        graph_embeddings: th.Tensor,
        non_decision_tabular_features: th.Tensor,
        obj_model: Callable[[Any, Any, Any], Any],
        params: Params,
        seed: Optional[int] = None,
    ) -> None:
        self.n_stages = n_stages
        self.graph_embeddings = graph_embeddings
        self.non_decision_tabular_features = non_decision_tabular_features
        self.seed = seed

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

    def solve(self) -> Tuple[np.ndarray, pd.DataFrame]:
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
                failed_flags = ["obj1", "obj2"]
                F = -1 * np.ones((1, 2))
                Theta = pd.DataFrame(-1, index=np.arange(1), columns=failed_flags)
                print("Timed out for query")
        else:
            F, Theta = self.div_moo_solve()

        return F, Theta

    def div_moo_solve(self) -> Tuple[np.ndarray, pd.DataFrame]:
        start_qs_tuning = time.time()
        f_df, conf_df = self.get_qs_set()
        print(f"time cost of getting effective set is {time.time() - start_qs_tuning}")

        start_dag_opt = time.time()
        F, Theta = self.dag_opt(f_df, conf_df)
        print(f"time cost of dag optimization is {time.time() - start_dag_opt}")
        return F, Theta

    def get_qs_set(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # 1. randomly sample each QS
        clustering_features = self.c_samples
        p_samples, s_samples = self.p_samples, self.s_samples

        # 2. cluster_based optimal theta_p and theta_p_s estimation

        (
            tuned_f_df,
            tuned_conf_df,
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
        union_opt_theta_c_list = self._union_opt_theta_c(tuned_f_df, tuned_conf_df)
        print(f"time cost of union theta_c is: {time.time() - start_union_c}")

        # 4. corssover enrichment: use crossover to enrich new \theta_c
        # location = 5
        start_extend_c = time.time()
        new_theta_c_list = self._extend_c(
            self.cross_location, union_opt_theta_c_list.copy()
        )
        print(f"time cost of extending c is {time.time() - start_extend_c}")

        # 5. cluster-based theta_p_s estimation
        start_predict = time.time()
        # predict labels
        pred_cluster_label = cluster_model.predict(new_theta_c_list)  # type: ignore
        uniq_cluster_label = np.unique(pred_cluster_label)
        # colume 1: cluster id; colume 2: new theta_c id
        new_label_arr = np.vstack(
            (pred_cluster_label, np.arange(pred_cluster_label.shape[0]))
        ).T
        new_cluster_members = [
            new_label_arr[:, 1][np.where(new_label_arr[:, 0] == label)].tolist()
            for label in uniq_cluster_label
        ]
        print(
            f"time cost of predicting cluster labels for new theta_c "
            f"is {time.time() - start_predict}"
        )
        start_est_opt_p_new_c = time.time()
        new_tuned_f_df, new_tuned_conf_df = self._estimate_opt_theta_p_s(
            th.Tensor(new_theta_c_list),
            uniq_cluster_label,
            new_cluster_members,
            label_rep_theta_c_mapping,
            tuned_conf_df,
            mode="estimate_new",
        )
        print(
            f"time cost of estimating optimal theta_p for new theta_c "
            f"is {time.time() - start_est_opt_p_new_c}"
        )
        # 6. union all solutions
        all_f_df = pd.concat([tuned_f_df, new_tuned_f_df])
        all_conf_df = pd.concat([tuned_conf_df, new_tuned_conf_df])

        return all_f_df, all_conf_df

    def dag_opt(
        self,
        f_df: pd.DataFrame,
        conf_df: pd.DataFrame,
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        dag_opt = DAGOpt(f_df, conf_df, self.dag_opt_algo, self.runmode)

        F, Theta = dag_opt.solve()
        return F, Theta

    # -------get QS effective set------
    def _cluster_based_estimate_opt_theta_p_s(
        self,
        n_clusters: int,
        clustering_features: th.Tensor,
        p_samples: th.Tensor,
        s_samples: th.Tensor,
        cluster_algo: str,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, object, Dict[Any, Any]]:
        # clustering theta_c
        clusters_labels, cluster_model = self._clustering_method(
            n_clusters, clustering_features, theta_c_cluster_algo=cluster_algo
        )
        (
            represent_theta_c,
            cluster_members,
            label_rep_theta_c_mapping,
        ) = self.clustering_representation(clusters_labels)

        start_tune_p = time.time()
        f_df, conf_df = self._tune_p(
            represent_theta_c,
            label_rep_theta_c_mapping,
            clustering_features,
            p_samples,
            s_samples,
            self.n_stages,
        )
        print(f"time cost of tuning theta_p is {time.time() - start_tune_p}")
        # estimate optimal theta_p and theta_s
        start_est_opt_p = time.time()
        tuned_f_df, tuned_conf_df = self._estimate_opt_theta_p_s(
            clustering_features,
            np.arange(len(clusters_labels)),
            cluster_members,
            label_rep_theta_c_mapping,
            conf_df,
        )
        print(
            f"time cost of estimating optimal theta_p for all theta_c "
            f"is {time.time() - start_est_opt_p}"
        )
        return tuned_f_df, tuned_conf_df, cluster_model, label_rep_theta_c_mapping

    def _clustering_method(
        self, n_cluster: int, cluster_features: th.Tensor, theta_c_cluster_algo: str
    ) -> Tuple[Any, object]:
        if theta_c_cluster_algo == "k-means":
            # print("k-means starts")
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
        clustering_features: th.Tensor,
        cluster_labels: np.ndarray,
        cluster_members: List[List[int]],
        label_rep_theta_c_mapping: Dict[Any, Any],
        conf_df: pd.DataFrame,
        mode: str = "estimate_exist",
        runmode: str = "multiprocessing",
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        len_theta_c = clustering_features.shape[1]

        final_f_df_list = []
        final_conf_df_list = []
        for stage_id in range(self.n_stages):
            # stage_values = conf_df.query(f"qs_id == {stage_id}")
            qs_f_df_list = []
            qs_conf_df_list = []
            for cluster_id, cm in zip(
                cluster_labels, cluster_members
            ):  # the number of clusters, i is the cluster index
                start_gen_theta = time.time()
                c_id = label_rep_theta_c_mapping[cluster_id]  # theta_c id
                opt_c_p_s = conf_df.query(f"qs_id == {stage_id}").query(
                    f"c_id == {c_id}"
                )

                # assign opt_p to all cluster members of theta_c
                len_cluster_member = len(cm)
                len_opt_p = opt_c_p_s.shape[0]

                mesh_theta_c = clustering_features[cm].repeat_interleave(
                    len_opt_p, dim=0
                )
                mesh_theta_p_s = th.Tensor(
                    opt_c_p_s.iloc[:, len_theta_c:].values
                ).repeat(len_cluster_member, 1)
                mesh_theta = th.cat([mesh_theta_c, mesh_theta_p_s], dim=1)

                mesh_graph_embeddings = (
                    self.graph_embeddings[stage_id]
                    .reshape(1, -1)
                    .repeat_interleave(mesh_theta.shape[0], dim=0)
                )
                mesh_non_decision_tabular_features = (
                    self.non_decision_tabular_features[stage_id]
                    .reshape(1, -1)
                    .repeat_interleave(mesh_theta.shape[0], dim=0)
                )

                # y_hat = self.obj_model(
                #     mesh_graph_embeddings,
                #     mesh_non_decision_tabular_features,
                #     mesh_theta,
                # )
                y_hat = self._get_obj_values(
                    mesh_theta.shape[0],
                    mesh_graph_embeddings,
                    mesh_non_decision_tabular_features,
                    mesh_theta,
                )
                print(
                    f"time cost of getting objective values of theta with "
                    f"opt theta_p of all theta_c is {time.time() - start_gen_theta}"
                )

                start_filter = time.time()
                # split_y_hat = y_hat.split(len_opt_p)
                split_y_hat = th.split(y_hat, len_opt_p)
                # split_conf = mesh_theta.split(len_opt_p)
                split_conf = th.split(mesh_theta, len_opt_p)
                assert len(split_conf) == len_cluster_member

                opt_f_df_list = []
                opt_conf_df_list = []
                for i, (sub_f, sub_conf) in enumerate(zip(split_y_hat, split_conf)):
                    po_ind = pg.non_dominated_front_2d(sub_f).tolist()
                    po_objs = sub_f[po_ind]
                    po_confs = sub_conf[po_ind]

                    if mode == "estimate_exist":
                        index = [
                            np.ones(
                                [
                                    len(po_ind),
                                ]
                            )
                            * stage_id,
                            np.ones(
                                [
                                    len(po_ind),
                                ]
                            )
                            * cm[i],
                        ]
                    else:
                        index = [
                            np.ones(
                                [
                                    len(po_ind),
                                ]
                            )
                            * stage_id,
                            np.ones(
                                [
                                    len(po_ind),
                                ]
                            )
                            * (cm[i] + self.c_samples.shape[0]),
                        ]
                    indices = pd.MultiIndex.from_tuples(
                        list(zip(*index)), names=["qs_id", "c_id"]
                    )
                    df_f = pd.DataFrame(po_objs, index=indices)
                    df_conf = pd.DataFrame(po_confs, index=indices)

                    opt_f_df_list.append(df_f)
                    opt_conf_df_list.append(df_conf)

                print(
                    f"time cost of filter dominated theta_p of each "
                    f"theta_c is {time.time() - start_filter}"
                )
                qs_f_df_list.append(pd.concat(opt_f_df_list))
                qs_conf_df_list.append(pd.concat(opt_conf_df_list))

            # check whether all the theta_c are assigned with optimal theta_p
            assert (
                np.unique(
                    np.array(pd.concat(qs_f_df_list).index.values.tolist())[:, 1]
                ).shape[0]
                == clustering_features.shape[0]
            )
            final_f_df_list.append(pd.concat(qs_f_df_list))
            final_conf_df_list.append(pd.concat(qs_conf_df_list))
        start_concat = time.time()
        result_f = pd.concat(final_f_df_list)
        result_conf = pd.concat(final_conf_df_list)
        print(
            f"time cost of concat f_df and conf_df list is {time.time() - start_concat}"
        )
        return result_f, result_conf

    def _union_opt_theta_c(
        self, tuned_f_df: pd.DataFrame, tuned_conf_df: pd.DataFrame
    ) -> List[List[Any]]:
        local_opt_theta_c_list = []
        for stage_id in range(self.n_stages):
            qs_values = tuned_f_df.query(f"qs_id == {stage_id}")
            po_ind = pg.non_dominated_front_2d(qs_values)
            # fixme: double-check iloc
            po_theta_c = tuned_conf_df.iloc[po_ind, :8].values.tolist()
            local_opt_theta_c_list.append(po_theta_c)

        return sum(local_opt_theta_c_list, [])

    def _extend_c(
        self,
        location: int,
        c_list: List[List[Any]],
    ) -> List[List[Any]]:
        # crossover to generate new \theta_c
        extend_c_list = c_list
        uniq_res_c = np.unique(np.array(c_list)[:, :location], axis=0)
        uniq_non_res_c = np.unique(np.array(c_list)[:, location:8], axis=0)

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
        # which are already included in the initial theta_c candidates
        uniq_new_c = np.unique(extend_theta_c, axis=0)
        matching_rows = np.all(
            uniq_new_c[:, None, :] == np.array(c_list)[None, :, :], axis=-1
        )
        matching_indices = np.where(matching_rows.any(axis=-1))[0]
        all_new_c_inds = np.arange(uniq_new_c.shape[0])
        mask = ~np.isin(all_new_c_inds, matching_indices)
        filtered_inds = all_new_c_inds[mask]

        extend_c_list.extend(extend_theta_c[filtered_inds].tolist())
        return extend_c_list

    def _tune_p(
        self,
        represent_theta_c: List[int],
        label_rep_theta_c_mapping: Dict[Any, Any],
        clustering_features: th.Tensor,
        p_samples: th.Tensor,
        s_samples: th.Tensor,
        n_stages: int,
        runmode: str = "multiprocessing",
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        rep_c_samples = clustering_features[represent_theta_c]
        n_evals = rep_c_samples.shape[0] * p_samples.shape[0] * n_stages
        mesh_c = rep_c_samples.repeat_interleave(p_samples.shape[0], dim=0).repeat(
            n_stages, 1
        )
        mesh_p = p_samples.repeat(rep_c_samples.shape[0], 1).repeat(n_stages, 1)
        mesh_s = s_samples.repeat(n_evals, 1)
        assert mesh_c.shape[0] == mesh_p.shape[0] and mesh_s.shape[0] == mesh_p.shape[0]
        mesh_theta = th.cat([mesh_c, mesh_p, mesh_s], dim=1)
        mesh_graph_embeddings = self.graph_embeddings.repeat_interleave(
            p_samples.shape[0] * rep_c_samples.shape[0], dim=0
        )
        mesh_non_decision_tabular_features = (
            self.non_decision_tabular_features.repeat_interleave(
                p_samples.shape[0] * rep_c_samples.shape[0], dim=0
            )
        )
        y_hat = self._get_obj_values(
            n_evals,
            mesh_graph_embeddings,
            mesh_non_decision_tabular_features,
            mesh_theta,
        )

        start_filter = time.time()
        split_y_hat = th.split(y_hat, p_samples.shape[0])
        split_theta = th.split(mesh_theta, p_samples.shape[0])

        df_f_list = []
        df_conf_list = []
        for i, (sub_f, sub_conf) in enumerate(zip(split_y_hat, split_theta)):
            stage_id = int(i / rep_c_samples.shape[0])
            c_id = label_rep_theta_c_mapping[int(i % rep_c_samples.shape[0])]
            po_ind = pg.non_dominated_front_2d(sub_f).tolist()
            po_objs = sub_f[po_ind]
            po_confs = sub_conf[po_ind]

            index = [
                np.ones(
                    [
                        len(po_ind),
                    ]
                )
                * stage_id,
                np.ones(
                    [
                        len(po_ind),
                    ]
                )
                * c_id,
            ]
            indices = pd.MultiIndex.from_tuples(
                list(zip(*index)), names=["qs_id", "c_id"]
            )
            df_f = pd.DataFrame(po_objs, index=indices)
            df_conf = pd.DataFrame(po_confs, index=indices)

            df_f_list.append(df_f)
            df_conf_list.append(df_conf)

        print(
            f"time cost of filtering dominated theta_p of "
            f"all theta_c is {time.time() - start_filter}"
        )

        return pd.concat(df_f_list), pd.concat(df_conf_list)

    def _get_obj_values(
        self,
        n_evals: int,
        mesh_graph_embeddings: th.Tensor,
        mesh_non_decision_tabular_features: th.Tensor,
        mesh_theta: th.Tensor,
    ) -> th.Tensor:
        if n_evals < 5120:
            loader = DataLoader(
                dataset=TensorDataset(
                    mesh_graph_embeddings,
                    mesh_non_decision_tabular_features,
                    mesh_theta,
                ),
                batch_size=512,
                shuffle=False,
                num_workers=0,
            )
        elif n_evals < 51200:
            loader = DataLoader(
                dataset=TensorDataset(
                    mesh_graph_embeddings,
                    mesh_non_decision_tabular_features,
                    mesh_theta,
                ),
                batch_size=2048,
                shuffle=False,
                num_workers=0,
            )
        elif n_evals < 512000:
            loader = DataLoader(
                dataset=TensorDataset(
                    mesh_graph_embeddings,
                    mesh_non_decision_tabular_features,
                    mesh_theta,
                ),
                batch_size=4096,
                shuffle=False,
                num_workers=0,
            )
        else:
            loader = DataLoader(
                dataset=TensorDataset(
                    mesh_graph_embeddings,
                    mesh_non_decision_tabular_features,
                    mesh_theta,
                ),
                batch_size=8092,
                shuffle=False,
                num_workers=0,
            )

        with th.no_grad():
            y_hat_list = []
            for graph_emb_batch, non_dec_batch, theta_batch in loader:
                y_hat_batch = self.obj_model(
                    graph_emb_batch, non_dec_batch, theta_batch
                )
                y_hat_list.append(y_hat_batch)
            y_hat = th.cat(y_hat_list)

        return y_hat

    def _tune_p_per_node(self) -> None:
        pass

    def _est_opt_p_per_node(self) -> None:
        pass
