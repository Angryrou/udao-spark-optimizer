import os
import time
from argparse import ArgumentParser
from pathlib import Path

from autogluon.tabular import TabularPredictor
from train_ensemble_models import get_ag_data

from udao_spark.model.mulitlabel_predictor import MultilabelPredictor  # type: ignore
from udao_spark.model.utils import wmape
from udao_spark.optimizer.utils import get_ag_meta
from udao_spark.utils.evaluation import get_ag_pred_objs
from udao_spark.utils.params import get_ag_parameters
from udao_trace.utils import JsonHandler


def get_parser() -> ArgumentParser:
    parser = get_ag_parameters()
    # fmt: off
    parser.add_argument("--ag_model_q_latency", type=str, default=None,
                        help="specific model name for AG for Q_R latency")
    parser.add_argument("--ag_model_q_io", type=str, default=None,
                        help="specific model name for AG for Q_R IO")
    parser.add_argument("--force", action="store_true",
                        help="Enable forcing running results")
    parser.add_argument("--bm_gtn_model", type=str, default=None,
                        help="gtn of the model pretrained.")
    # fmt: on
    return parser


if __name__ == "__main__":
    params = get_parser().parse_args()

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
        params.fold,
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

    ag_path = ag_meta["ag_path"] + "_plus_tpl" + "/"
    if os.path.exists(ag_path):
        predictor = MultilabelPredictor.load(f"{ag_path}")
        print("model found and loadable at", ag_path)
        ag_model = {
            "latency_s": params.ag_model_q_latency,
            "io_mb": params.ag_model_q_io,
        }
        bm_target = params.bm_gtn_model or bm
        objs_true, objs_pred, dt_s, throughput, metrics = get_ag_pred_objs(
            base_dir,
            bm,
            q_type,
            debug,
            graph_choice,
            split="test",
            ag_meta=ag_meta,
            fold=params.fold,
            force=params.force,
            ag_model=ag_model,
            bm_target=bm_target,
            xfer_gtn_only=True,
            plus_tpl=True,
        )
        print(f"metrics: {metrics}, throughput (regr only): {throughput} K/s")

        if q_type.startswith("qs"):
            m1, m2 = metrics["ana_latency_s"], metrics["io_mb"]
        else:
            m1, m2 = metrics["latency_s"], metrics["io_mb"]
        print("-" * 20)
        print(
            "{:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & "
            "{:.3f} & {:.3f} & {:.3f} & {:.0f} \\\\".format(
                m1["wmape"],
                m1["p50_wape"],
                m1["p90_wape"],
                m1["corr"],
                m2["wmape"],
                m2["p50_wape"],
                m2["p90_wape"],
                m2["corr"],
                throughput,
            )
        )
        print()

    else:
        objectives = ag_data["objectives"]
        start_time_step1 = time.perf_counter_ns()
        predictor = MultilabelPredictor(
            path=ag_path,
            labels=objectives,
            problem_types=["regression"] * len(objectives),
            eval_metrics=[wmape] * len(objectives),
            consider_labels_correlation=False,
        )
        time_dict_step1 = predictor.fit(
            train_data=train_data,
            excluded_model_types=["KNN"],
            tuning_data=val_data,
            presets_lat=ag_sign,
            presets_io=ag_sign,  # we used to force this to be medium quality because
            # Shuffle Size / IO tends to have a much smaller error and we can save
            # inference time by using a lower quality model for IO
            use_bag_holdout=True,
            infer_limit=infer_limit,
            infer_limit_batch_size=infer_limit_batch_size,
            num_gpus=num_gpus,
            time_limit=None if time_limit is None else time_limit // len(objectives),
            ds_args={"memory_safe_fits": False},
            # set False to avoid memory issues when training io with high quality
        )
        dt1 = (time.perf_counter_ns() - start_time_step1) / 1e9

        print("step 2: .fit_weighted_ensemble()")
        start_time_step2 = time.perf_counter_ns()
        time_dict_step2 = {}
        for obj in predictor.predictors.keys():
            add_start_time = time.perf_counter_ns()
            models = predictor.get_predictor(obj).model_names(stack_name="core")
            return_models_po = predictor.get_predictor(obj).fit_weighted_ensemble(
                expand_pareto_frontier=True,
                name_suffix="PO",
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

            time_dict_step2[obj] = (time.perf_counter_ns() - add_start_time) / 1e9

            print(f"get po-models from {obj}: {return_models_po}")
            print(f"get fast-po-models from {obj}: {return_models_fast_po}")
            print(
                f"ensemble models for {obj} including "
                f"{predictor.get_predictor(obj).model_names()}"
            )
        dt2 = (time.perf_counter_ns() - start_time_step2) / 1e9

        print(f"dt1: {dt1:.0f} s, dt2: {dt2:.0f} s")
        print("saving runtime to", ag_path)

        time_dict = {
            "dt1_s": dt1,
            "dt1_s_per_obj": time_dict_step1,
            "dt2_s": dt2,
            "dt2_s_per_obj": time_dict_step2,
        }
        JsonHandler.dump_to_file(time_dict, f"{ag_path}/runtime.json")
