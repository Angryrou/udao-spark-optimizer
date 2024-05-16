import json
import os
import time
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch as th
from udao.optimization.concepts.utils import InputParameters, InputVariables

from udao_trace.utils import JsonHandler

from ..model.utils import get_graph_ckp_info
from ..utils.constants import EPS
from ..utils.logging import logger
from ..utils.params import QType

UdaoNumeric = Union[float, pd.Series, np.ndarray, th.Tensor]

aws_cost_cpu_hour_ratio = 0.052624
aws_cost_mem_hour_ratio = 0.0057785  # for GB*H
io_gb_ratio = 0.02  # for GB


def get_cloud_cost_wo_io(
    lat: UdaoNumeric, cores: UdaoNumeric, mem: UdaoNumeric, nexec: UdaoNumeric
) -> UdaoNumeric:
    cpu_hour = (nexec + 1) * cores * lat / 3600.0
    mem_hour = (nexec + 1) * mem * lat / 3600.0
    cost = cpu_hour * aws_cost_cpu_hour_ratio + mem_hour * aws_cost_mem_hour_ratio
    return cost


def get_cloud_cost_add_io(
    cost_wo_io: UdaoNumeric,
    io_mb: UdaoNumeric,
) -> UdaoNumeric:
    return cost_wo_io + io_mb / 1024.0 * io_gb_ratio


def get_cloud_cost_w_io(
    lat: UdaoNumeric,
    cores: UdaoNumeric,
    mem: UdaoNumeric,
    nexec: UdaoNumeric,
    io_mb: UdaoNumeric,
) -> UdaoNumeric:
    return get_cloud_cost_add_io(get_cloud_cost_wo_io(lat, cores, mem, nexec), io_mb)


def get_weights_path_dict(
    bm: str, hp_choice: str, graph_choice: str, q_type: QType
) -> str:
    weights_cache = JsonHandler.load_json("assets/mlp_configs.json")
    try:
        weights_path = weights_cache[bm][hp_choice][graph_choice][q_type]
    except KeyError:
        raise Exception(
            f"weights_path not found for {bm}/{hp_choice}/{graph_choice}/{q_type}"
        )
    return weights_path


def get_ag_meta(
    bm: str,
    hp_choice: str,
    graph_choice: str,
    q_type: QType,
    ag_sign: str,
    infer_limit: Optional[float],
    infer_limit_batch_size: Optional[int],
    time_limit: Optional[int],
) -> Dict[str, str]:
    graph_weights_path = get_weights_path_dict(bm, hp_choice, graph_choice, q_type)
    ag_prefix, model_sign, model_params_path, data_processor_path = get_graph_ckp_info(
        graph_weights_path
    )
    if infer_limit is None:
        ag_full_name = f"{ag_sign}_{hp_choice}"
    else:
        if infer_limit_batch_size is None:
            infer_limit_batch_size = 10000
        ag_full_name = "{}_{}_infer_limit_{}_batch_size_{}".format(
            ag_sign, hp_choice, infer_limit, infer_limit_batch_size
        )

    if time_limit is not None:
        ag_full_name += f"_time_limit_{time_limit}s"

    ag_path = "AutogluonModels/{}/{}/{}/{}".format(
        ag_prefix, q_type, graph_choice, ag_full_name
    )
    return {
        "ag_path": ag_path,
        "graph_weights_path": graph_weights_path,
        "model_sign": model_sign,
        "model_params_path": model_params_path,
        "data_processor_path": data_processor_path,
        "ag_full_name": ag_full_name,
    }


def save_results(path: str, results: np.ndarray, mode: str = "data") -> None:
    file_path = path
    if not os.path.exists(file_path):
        os.makedirs(file_path, exist_ok=True)

    if "Theta" in mode or "query_id" in mode:
        np.savetxt(
            f"{file_path}/{mode}.txt", results, delimiter=" ", newline="\n", fmt="%s"
        )
    else:
        np.savetxt(f"{file_path}/{mode}.txt", results)


def save_json(save_json_path: str, data: dict, mode: str = "time_dict") -> None:
    if not os.path.exists(save_json_path):
        os.makedirs(save_json_path, exist_ok=True)

    with open(f"{save_json_path}/{mode}.json", "w") as fp:
        json.dump(data, fp, indent=4, separators=(",", ": "))


# a quite efficient way to get the indexes of pareto points
# https://stackoverflow.com/a/40239615
def is_pareto_efficient(costs: np.ndarray, return_mask: bool = True) -> np.ndarray:
    ## reuse code in VLDB2022
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index < len(costs):
        nondominated_point_mask = np.any(costs < costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index]) + 1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype=bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient


def keep_non_dominated(
    po_obj_list: List[np.ndarray], po_var_list: List[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    ## reuse code in VLDB2022
    assert len(po_obj_list) == len(po_var_list)
    if len(po_obj_list) == 0:
        return np.array([-1]), np.array([-1])
    elif len(po_obj_list) == 1:
        return np.array(po_obj_list), np.array(po_var_list)
    else:
        po_objs_cand = np.array(po_obj_list)
        po_vars_cand = np.array(po_var_list)
        po_inds = is_pareto_efficient(po_objs_cand)
        po_objs = po_objs_cand[po_inds]
        po_vars = po_vars_cand[po_inds]
        return po_objs, po_vars


def even_weights(stepsize: float, m: int) -> List[Any]:
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

    assert all(np.round(np.sum(ws_pairs, axis=1), 10) == 1)
    return ws_pairs


def weighted_utopia_nearest(
    pareto_objs: np.ndarray,
    pareto_confs: np.ndarray,
    weights: np.ndarray = np.array([1, 1]),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    return the Pareto point that is closest to the utopia point
    in a weighted distance function
    """
    n_pareto = pareto_objs.shape[0]
    if n_pareto == 0:
        logger.error("No Pareto point, return None")
        raise ValueError("No Pareto point, return None")
    if n_pareto == 1:
        # (2,), (n, 2)
        return pareto_objs[0], pareto_confs[0]

    utopia = np.zeros_like(pareto_objs[0])
    min_objs, max_objs = pareto_objs.min(0), pareto_objs.max(0)
    pareto_norm = (pareto_objs - min_objs) / (max_objs - min_objs)
    pareto_weighted_norm = pareto_norm * weights
    dists = np.sum((pareto_weighted_norm - utopia) ** 2, axis=1)
    wun_id = np.argmin(dists)

    picked_pareto = pareto_objs[wun_id]
    picked_confs = pareto_confs[wun_id]

    return picked_pareto, picked_confs


def utopia_nearest(
    po_objs: np.ndarray,
    po_confs: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    return the Pareto point that is closest to the utopia point
    """
    if po_objs.ndim != 2 or po_confs.ndim != 2:
        raise ValueError("po_objs and po_confs should be 2D arrays")
    if len(po_objs) == 0:
        logger.error("No Pareto point, return None")
        return None, None
    if len(po_objs) == 1:
        logger.warning("Only one Pareto point, return it directly")
        return po_objs[0], po_confs[0]
    if po_objs.dtype != np.float32:
        po_objs = po_objs.astype(np.float32)

    objs_min, objs_max = po_objs.min(axis=0), po_objs.max(axis=0)
    objs_range = objs_max - objs_min
    objs_range[np.where(objs_range == 0)] = EPS
    po_objs_norm = (po_objs - objs_min) / objs_range
    # after normalization, the utopia point locates at [0, 0]
    distances = np.linalg.norm(po_objs_norm, axis=1)
    un_ind = np.argmin(distances)
    return po_objs[un_ind], po_confs[un_ind]


class Model:
    def __init__(
        self,
        n_stages: int,
        len_theta_c: int,
        len_theta_p: int,
        len_theta_s: int,
        graph_embeddings: th.Tensor,
        non_decision_tabular_features: Union[th.Tensor, pd.DataFrame],
        obj_model: Callable[[Any, Any, Any, Any], Any],
        use_ag: bool,
        ag_model: dict[str, str],
        is_query_control: bool,
    ):
        self.n_stages = n_stages
        self.len_theta_c = len_theta_c
        self.len_theta_p = len_theta_p
        self.len_theta_s = len_theta_s
        self.graph_embeddings = graph_embeddings
        self.non_decision_tabular_features = non_decision_tabular_features
        self.obj_model = obj_model
        self.use_ag = use_ag
        self.ag_model = ag_model
        self.is_query_control = is_query_control

    def Obj1(
        self, input_variables: InputVariables, input_parameters: InputParameters = None
    ) -> th.Tensor:
        theta = th.vstack(list(input_variables.values())).T
        n_repeat = theta.shape[0]
        mesh_theta_c: Union[np.ndarray, th.Tensor]
        theta_p_s: Union[np.ndarray, th.Tensor]
        mesh_tabular_features: Union[th.Tensor, pd.DataFrame]

        theta = th.vstack(list(input_variables.values())).T
        # shape: (n_samples/grids * n_objs)
        mesh_theta_c = theta[:, : self.len_theta_c].repeat(self.n_stages, 1)
        len_theta_p_s = self.len_theta_p + self.len_theta_s

        if self.is_query_control:
            assert theta.shape[1] == (
                self.len_theta_c + self.len_theta_p + self.len_theta_s
            )
            theta_p_s = theta[:, self.len_theta_c :].repeat(self.n_stages, 1)
        else:
            assert theta.shape[1] == (self.len_theta_c + self.n_stages * len_theta_p_s)
            theta_p_s = theta[:, self.len_theta_c :].reshape(
                self.n_stages * n_repeat, len_theta_p_s
            )
        theta_all = th.cat([mesh_theta_c, theta_p_s], dim=1)
        mesh_graph_embeddings = self.graph_embeddings.repeat_interleave(n_repeat, dim=0)

        if self.use_ag:
            assert isinstance(self.non_decision_tabular_features, pd.DataFrame)
            mesh_tabular_features = self.non_decision_tabular_features.loc[
                np.repeat(self.non_decision_tabular_features.index, n_repeat)
            ].reset_index(drop=True)
            objs = self.obj_model(
                mesh_graph_embeddings.cpu().numpy(),
                mesh_tabular_features,
                theta_all.cpu().numpy(),
                self.ag_model,
            )
        else:
            assert isinstance(self.non_decision_tabular_features, th.Tensor)
            mesh_tabular_features = (
                self.non_decision_tabular_features.repeat_interleave(n_repeat, dim=0)
            )
            objs = self.obj_model(
                mesh_graph_embeddings,
                mesh_tabular_features,
                theta_all.type(th.float32),
                "",
            )
        # shape (n_samples, n_stages)
        latency = np.vstack(np.split(objs[:, 0], self.n_stages)).T
        cost = np.vstack(np.split(objs[:, 1], self.n_stages)).T

        query_latency = np.sum(latency, axis=1)
        query_cost = np.sum(cost, axis=1)
        query_objs = np.vstack((query_latency, query_cost)).T
        y = query_objs[:, 0]
        return th.reshape(th.tensor(y, dtype=th.float32), (-1, 1))

    def Obj2(
        self, input_variables: InputVariables, input_parameters: InputParameters = None
    ) -> th.Tensor:
        theta = th.vstack(list(input_variables.values())).T
        n_repeat = theta.shape[0]
        mesh_theta_c: Union[np.ndarray, th.Tensor]
        theta_p_s: Union[np.ndarray, th.Tensor]
        mesh_tabular_features: Union[th.Tensor, pd.DataFrame]

        theta = th.vstack(list(input_variables.values())).T
        # shape: (n_samples/grids * n_objs)
        mesh_theta_c = theta[:, : self.len_theta_c].repeat(self.n_stages, 1)
        len_theta_p_s = self.len_theta_p + self.len_theta_s
        if self.is_query_control:
            theta_p_s = theta[:, self.len_theta_c :].repeat(self.n_stages, 1)
        else:
            theta_p_s = theta[:, self.len_theta_c :].reshape(
                self.n_stages * n_repeat, len_theta_p_s
            )
        theta_all = th.cat([mesh_theta_c, theta_p_s], dim=1)
        mesh_graph_embeddings = self.graph_embeddings.repeat_interleave(n_repeat, dim=0)

        if self.use_ag:
            assert isinstance(self.non_decision_tabular_features, pd.DataFrame)
            mesh_tabular_features = self.non_decision_tabular_features.loc[
                np.repeat(self.non_decision_tabular_features.index, n_repeat)
            ].reset_index(drop=True)
            objs = self.obj_model(
                mesh_graph_embeddings.cpu().numpy(),
                mesh_tabular_features,
                theta_all.cpu().numpy(),
                self.ag_model,
            )
        else:
            assert isinstance(self.non_decision_tabular_features, th.Tensor)
            mesh_tabular_features = (
                self.non_decision_tabular_features.repeat_interleave(n_repeat, dim=0)
            )
            objs = self.obj_model(
                mesh_graph_embeddings,
                mesh_tabular_features,
                theta_all.type(th.float32),
                "",
            )
        # shape (n_samples, n_stages)
        latency = np.vstack(np.split(objs[:, 0], self.n_stages)).T
        cost = np.vstack(np.split(objs[:, 1], self.n_stages)).T

        query_latency = np.sum(latency, axis=1)
        query_cost = np.sum(cost, axis=1)
        query_objs = np.vstack((query_latency, query_cost)).T
        y = query_objs[:, 1]
        return th.reshape(th.tensor(y, dtype=th.float32), (-1, 1))


def timeis(f: Callable[..., Any]) -> Callable[..., Tuple[Any, float]]:
    @wraps(f)
    def wrap(*args: Any, **kw: Any) -> Tuple[Any, float]:
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        time_cost = te - ts
        return result, time_cost

    return wrap
