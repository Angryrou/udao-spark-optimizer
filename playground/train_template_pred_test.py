import os
from argparse import ArgumentParser
from pathlib import Path

from autogluon.tabular import TabularPredictor
from train_ensemble_models import get_ag_data

from udao_spark.optimizer.utils import get_ag_meta
from udao_spark.utils.params import get_ag_parameters
from udao_trace.utils import JsonHandler


def get_parser() -> ArgumentParser:
    parser = get_ag_parameters()
    # fmt: off
    parser.add_argument("--im-only", action="store_true",)
    parser.add_argument("--conf-only", action="store_true",)
    # fmt: on
    return parser


if __name__ == "__main__":
    params = get_parser().parse_args()

    if params.im_only and params.conf_only:
        raise ValueError("Cannot specify both im-only and conf-only")

    base_dir = Path(__file__).parent
    bm, q_type, debug = params.benchmark, params.q_type, params.debug
    hp_choice, graph_choice = params.hp_choice, params.graph_choice
    num_gpus, ag_sign = params.num_gpus, params.ag_sign
    infer_limit = params.infer_limit
    infer_limit_batch_size = params.infer_limit_batch_size
    time_limit = params.ag_time_limit
    if graph_choice != "none":
        raise ValueError("graph_choice must be none")
    if q_type != "q_compile":
        raise ValueError(f"Diagnosing {q_type} is not our focus.")
    ag_meta = get_ag_meta(
        bm,
        "tuned-0215",
        graph_choice,
        q_type,
        ag_sign,
        infer_limit,
        infer_limit_batch_size,
        time_limit,
    )
    ag_data = get_ag_data(
        base_dir, bm, q_type, False, graph_choice, None, fold=params.fold
    )
    data_dict = {
        sp: da.join(daq[["template"]].astype("str"))
        for sp, da, daq in zip(
            ["train", "val", "test"], ag_data["data"], ag_data["data_queries"]
        )
    }
    train_data, val_data, test_data = (
        data_dict["train"],
        data_dict["val"],
        data_dict["test"],
    )

    path = ag_meta["ag_path"] + "_pred_tpl"
    if params.im_only:
        path += "_im"
        drop_columns = [
            c for c in train_data.columns if c.startswith("k") or c.startswith("s")
        ]
        train_data, val_data, test_data = (
            train_data.drop(columns=drop_columns),
            val_data.drop(columns=drop_columns),
            test_data.drop(columns=drop_columns),
        )
    if params.conf_only:
        path += "_conf"
        drop_columns = [c for c in train_data.columns if c.startswith("IM")]
        train_data, val_data, test_data = (
            train_data.drop(columns=drop_columns),
            val_data.drop(columns=drop_columns),
            test_data.drop(columns=drop_columns),
        )

    if os.path.exists(path):
        tpl_predictor = TabularPredictor.load(path)
        print("model found and loadable at", path)
    else:
        tpl_predictor = TabularPredictor(
            label="template", problem_type="multiclass", path=path
        )
        tpl_predictor.fit(
            train_data.drop(columns=["latency_s", "io_mb"]),
            tuning_data=val_data.drop(columns=["latency_s", "io_mb"]),
            presets=ag_sign,
            use_bag_holdout=True,
            infer_limit=infer_limit,
            infer_limit_batch_size=infer_limit_batch_size,
            num_gpus=num_gpus,
            time_limit=time_limit,
            ds_args={"memory_safe_fits": False},
        )
        print("model trained and saved at", path)
        res = tpl_predictor.evaluate(test_data, silent=True)
        JsonHandler.dump_to_file(res, f"{path}/eval.json")
