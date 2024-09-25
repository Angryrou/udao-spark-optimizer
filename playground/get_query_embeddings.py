import glob
import os
from typing import Dict, Optional, Tuple

from udao_spark.model.model_server import ModelServer
from udao_spark.utils.evaluation import get_graph_embedding
from udao_trace.utils import JsonHandler, PickleHandler


def get_pkl(fullpath: str) -> object:
    return PickleHandler.load(
        os.path.dirname(fullpath),
        os.path.basename(fullpath),
    )


def get_best_path(
    header: str = "cache_and_ckp/tpcds_102x490/q_compile",
    name: str = "ea0378f56dcf",
    graph_choice: str = "gtn_sk_mlp",
    target_op_feat: str = "type+cbo+op_enc+bitmap",
    suffix: str = "_1_0",
) -> str:
    ret_list = []
    struct_paths = glob.glob(
        f"{header}/{name}/*{graph_choice}*/model_struct_params.json"
    )
    for p in struct_paths:
        d = JsonHandler.load_json(p)
        op = "+".join([oo for oo in d["op_groups"] if oo != "\\"])
        if op == target_op_feat:
            cand_paths = glob.glob(
                f"{os.path.dirname(p)}/learning*{suffix}/obj_df_val*cpu.pkl"
            )
            val_loss = []
            for cand_path in cand_paths:
                res = get_pkl(cand_path)
                if not isinstance(res, Dict):
                    raise TypeError(f"res is not a dict: {res}")
                wmape_lat = res["metrics"]["latency_s"]["wmape"]
                wmape_io = res["metrics"]["io_mb"]["wmape"]
                val_loss.append(wmape_lat + wmape_io)
            min_index = val_loss.index(min(val_loss))
            best_path = cand_paths[min_index]
            # print(p)
            # print(best_path)
            # print()
            ret_list.append(os.path.dirname(best_path))
    if len(ret_list) != 1:
        raise ValueError(ret_list)
    return ret_list[0]


def get_query_emb_q_compile(
    header: str = "cache_and_ckp/tpcds_102x490/q_compile",
    name: str = "ea0378f56dcf",
    fold: Optional[int] = None,
    model_sign: str = "graph_gtn_sk_mlp",
) -> Tuple[Dict, Dict]:
    if fold is not None:
        name = f"{name}-{fold}"
    ckp_header = get_best_path(header, name)
    model_params_path = os.path.dirname(ckp_header) + "/model_struct_params.json"
    ckp_weight_path = glob.glob(f"{ckp_header}/*.ckpt")[0]
    ms = ModelServer.from_ckp_path(
        model_sign="graph_gtn_sk_mlp",
        model_params_path=model_params_path,
        weights_path=ckp_weight_path,
    )
    if model_sign not in [
        "graph_gtn_sk_mlp",
    ]:
        raise NotImplementedError("check whether split iterators need to be reorg.")

    split_iterators = PickleHandler.load(f"{header}/{name}", "split_iterators.pkl")
    if not isinstance(split_iterators, Dict):
        raise TypeError("split_iterators not found or not a desired type")
    index_splits_name = (
        "index_splits_q_compile.pkl"
        if fold is None
        else f"index_splits_q_compile-{fold}.pkl"
    )
    index_splits = PickleHandler.load(os.path.dirname(header), index_splits_name)
    if not isinstance(index_splits, Dict):
        raise TypeError(f"index_splits is not a dict: {index_splits}")

    graph_np_dict = get_graph_embedding(
        ms,
        split_iterators,
        index_splits,
        ckp_header,
        name="graph_np_dict.pkl",
    )
    return index_splits, graph_np_dict


get_query_emb_q_compile()
for fold in range(1, 11):
    get_query_emb_q_compile(fold=fold)
