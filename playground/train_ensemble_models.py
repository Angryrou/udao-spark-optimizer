import os.path
import time
from argparse import ArgumentParser
from pathlib import Path

from udao_spark.model.mulitlabel_predictor import MultilabelPredictor  # type: ignore
from udao_spark.model.utils import wmape
from udao_spark.optimizer.utils import get_ag_meta
from udao_spark.utils.evaluation import get_ag_data
from udao_spark.utils.params import get_ag_parameters
from udao_trace.utils import JsonHandler


def get_params() -> ArgumentParser:
    parser = get_ag_parameters()
    # fmt: off
    parser.add_argument("--recording", action="store_true",
                        help="Record the time for evaluation")
    # fmt: on
    return parser


if __name__ == "__main__":
    params = get_params().parse_args()
    bm, q_type, debug = params.benchmark, params.q_type, params.debug
    hp_choice, graph_choice = params.hp_choice, params.graph_choice
    num_gpus, ag_sign = params.num_gpus, params.ag_sign
    infer_limit = params.infer_limit
    infer_limit_batch_size = params.infer_limit_batch_size
    time_limit = params.ag_time_limit
    recording = params.recording
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
    if not recording:
        ag_path = ag_meta["ag_path"] + "/"
    else:
        ag_path = ag_meta["ag_path"] + "_recording/"

    ret = get_ag_data(base_dir, bm, q_type, debug, graph_choice, weights_path)
    train_data, val_data, test_data = ret["data"]
    ta, pw, objectives = ret["ta"], ret["pw"], ret["objectives"]
    if q_type.startswith("qs_"):
        objectives = list(filter(lambda x: x != "latency_s", objectives))
        train_data.drop(columns=["latency_s"], inplace=True)
        val_data.drop(columns=["latency_s"], inplace=True)
        test_data.drop(columns=["latency_s"], inplace=True)
    print("selected features:", train_data.columns)

    start_time = time.perf_counter_ns()
    if os.path.exists(ag_path):
        if recording:
            raise Exception("recording path already exists")
        predictor = MultilabelPredictor.load(f"{ag_path}")
        print("loaded predictor from", ag_path)
    else:
        print("not found, fitting")
        predictor = MultilabelPredictor(
            path=ag_path,
            labels=objectives,
            problem_types=["regression"] * len(objectives),
            eval_metrics=[wmape] * len(objectives),
            consider_labels_correlation=False,
        )
        predictor.fit(
            train_data=train_data,
            # num_stack_levels=1,
            # num_bag_folds=4,
            # hyperparameters={
            #     "NN_TORCH": {},
            #     "GBM": {},
            #     "CAT": {},
            #     "XGB": {},
            #     "FASTAI": {},
            #     "RF": [
            #         {
            #             "criterion": "gini",
            #             "ag_args": {
            #                 "name_suffix": "Gini",
            #                 "problem_types": ["binary", "multiclass"],
            #             },
            #         },
            #         {
            #             "criterion": "entropy",
            #             "ag_args": {
            #                 "name_suffix": "Entr",
            #                 "problem_types": ["binary", "multiclass"],
            #             },
            #         },
            #         {
            #             "criterion": "squared_error",
            #             "ag_args": {
            #                 "name_suffix": "MSE",
            #                 "problem_types": ["regression", "quantile"],
            #             },
            #         },
            #     ],
            # },
            excluded_model_types=["KNN"],
            tuning_data=val_data,
            presets_lat=ag_sign.split(","),
            presets_io=None,
            use_bag_holdout=True,
            infer_limit=infer_limit,
            infer_limit_batch_size=infer_limit_batch_size,
            num_gpus=num_gpus,
            time_limit=time_limit,
        )
    dt1 = time.perf_counter_ns() - start_time

    for obj in predictor.predictors.keys():
        models = predictor.get_predictor(obj).model_names(stack_name="core")
        return_models_po = predictor.get_predictor(obj).fit_weighted_ensemble(
            expand_pareto_frontier=True,
            name_suffix="PO",
        )
        predictor.get_predictor(obj).fit_weighted_ensemble(
            base_models=[
                m
                for m in models
                if "Large" not in m and "XT" not in m and "ExtraTree" not in m
            ],
            name_suffix="Fast",
        )
        return_models_fast_po = predictor.get_predictor(obj).fit_weighted_ensemble(
            base_models=[
                m
                for m in models
                if "Large" not in m and "XT" not in m and "ExtraTree" not in m
            ],
            expand_pareto_frontier=True,
            name_suffix="FastPO",
        )

        print(f"get po-models from {obj}: {return_models_po}")
        print(f"get fast-po-models from {obj}: {return_models_fast_po}")
        print(
            f"ensemble models for {obj} including "
            f"{predictor.get_predictor(obj).model_names()}"
        )

    dt2 = time.perf_counter_ns() - start_time

    print(f"dt1: {dt1 / 1e9} s, dt2: {dt2 / 1e9} s")
    if recording:
        print("saving runtime to", ag_path)
        JsonHandler.dump_to_file(
            {"dt1_s": dt1, "dt2_s": dt2}, f"{ag_path}/runtime.json"
        )
