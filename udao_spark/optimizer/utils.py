import os
from typing import Any, List, Tuple, Union

import numpy as np
import pandas as pd
import torch as th

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


def save_results(path: str, results: np.ndarray, mode: str = "data") -> None:
    file_path = path
    if not os.path.exists(file_path):
        os.makedirs(file_path, exist_ok=True)

    if mode == "Theta":
        np.savetxt(
            f"{file_path}/{mode}.txt", results, delimiter=" ", newline="\n", fmt="%s"
        )
    else:
        np.savetxt(f"{file_path}/{mode}.txt", results)


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


def _summarize_ret(
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
