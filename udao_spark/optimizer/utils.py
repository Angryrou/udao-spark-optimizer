from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
import torch as th

from udao_trace.utils import JsonHandler

from ..model.utils import get_graph_ckp_info
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
