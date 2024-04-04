import os.path
from pathlib import Path

from udao_spark.model.mulitlabel_predictor import MultilabelPredictor  # type: ignore
from udao_spark.optimizer.utils import get_ag_meta
from udao_spark.utils.evaluation import get_ag_data, get_ag_pred_objs
from udao_spark.utils.params import get_ag_parameters

if __name__ == "__main__":
    params = get_ag_parameters().parse_args()

    if params.q_type != "q_compile":
        raise ValueError(f"Diagnosing {params.q_type} is not our focus.")

    bm, q_type, debug = params.benchmark, params.q_type, params.debug
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
    ag_path = ag_meta["ag_path"] + "/"

    ret = get_ag_data(base_dir, bm, q_type, debug, graph_choice, weights_path)
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

    ag_model = {
        "ana_latency_s": params.ag_model_q_latency,
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
        force=False,
        ag_model=ag_model,
    )
