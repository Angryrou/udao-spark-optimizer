import os.path
from argparse import ArgumentParser
from pathlib import Path

from udao_spark.model.mulitlabel_predictor import MultilabelPredictor  # type: ignore
from udao_spark.optimizer.utils import get_ag_meta
from udao_spark.utils.evaluation import get_ag_data, get_ag_pred_objs
from udao_spark.utils.params import get_ag_parameters


def get_parser() -> ArgumentParser:
    parser = get_ag_parameters()
    # fmt: off
    parser.add_argument("--ag_model_q_latency", type=str, default=None,
                        help="specific model name for AG for Q_R latency")
    parser.add_argument("--ag_model_q_io", type=str, default=None,
                        help="specific model name for AG for Q_R IO")
    parser.add_argument("--bm_target", type=str, default="tpcds",
                        help="specific database for evaluation")
    parser.add_argument("--force", action="store_true",
                        help="Enable forcing running results")
    parser.add_argument("--new_recording", action="store_true",
                        help="Recording the breakdown training times")
    # fmt: on
    return parser


if __name__ == "__main__":
    params = get_parser().parse_args()

    # if params.q_type != "q_compile":
    #     raise ValueError(f"Diagnosing {params.q_type} is not our focus.")

    bm, q_type, debug = params.benchmark, params.q_type, params.debug
    bm_target = params.bm_target
    hp_choice, graph_choice = params.hp_choice, params.graph_choice
    num_gpus, ag_sign = params.num_gpus, params.ag_sign
    infer_limit = params.infer_limit
    infer_limit_batch_size = params.infer_limit_batch_size
    time_limit = params.ag_time_limit
    base_dir = Path(__file__).parent
    ag_meta = get_ag_meta(
        bm,
        hp_choice,
        graph_choice,
        q_type,
        ag_sign,
        infer_limit,
        infer_limit_batch_size,
        time_limit,
    )
    weights_path = ag_meta["graph_weights_path"]
    if params.new_recording:
        ag_meta["ag_path"] = ag_meta["ag_path"] + "new_recording"
    ag_path = ag_meta["ag_path"] + "/"

    ret = get_ag_data(
        base_dir, bm, q_type, debug, graph_choice, weights_path, bm_target=bm_target
    )
    train_data, val_data, test_data = ret["data"]
    ta, pw, objectives = ret["ta"], ret["pw"], ret["objectives"]
    if q_type.startswith("qs_"):
        objectives = list(filter(lambda x: x != "latency_s", objectives))
        train_data.drop(columns=["latency_s"], inplace=True)
        val_data.drop(columns=["latency_s"], inplace=True)
        test_data.drop(columns=["latency_s"], inplace=True)
    print("selected features:", train_data.columns)

    if os.path.exists(ag_path):
        predictor = MultilabelPredictor.load(f"{ag_path}")
        print("loaded predictor from", ag_path)
    else:
        raise Exception("run train_ensemble_models.py first")

    lat_obj_name = "ana_latency_s" if q_type.startswith("qs") else "latency_s"
    ag_model = {
        lat_obj_name: params.ag_model_q_latency,
        "io_mb": params.ag_model_q_io,
    }

    objs_true, objs_pred, dt_s, throughput, metrics = get_ag_pred_objs(
        base_dir,
        bm,
        q_type,
        debug,
        graph_choice,
        split="test",
        ag_meta=ag_meta,
        force=params.force,
        ag_model=ag_model,
        bm_target=bm_target,
    )

    print(f"metrics: {metrics}, throughput (regr only): {throughput} K/s")
