import os.path
from pathlib import Path

from autogluon.common.utils.utils import setup_outputdir

from playground.monotone_internal_autogluon import MonotonePredictor
from playground.monotone_network_autogluon import MonotoneTabularNeuralNetTorchModel, MonotoneXGBoostModel, \
    MonotoneCatBoostModel
from udao_spark.data.utils import get_ag_data
from udao_spark.model.mulitlabel_predictor import MultilabelPredictor  # type: ignore
from udao_spark.model.utils import wmape
from udao_spark.optimizer.utils import get_ag_meta
from udao_spark.utils.params import get_ag_parameters


class MonotoneMultilabelPredictor(MultilabelPredictor):
    def __init__(
            self,
            labels,
            path=None,
            problem_types=None,
            eval_metrics=None,
            consider_labels_correlation=True,
            **kwargs,
    ):
        if len(labels) < 2:
            raise ValueError(
                "MultilabelPredictor is only intended for predicting MULTIPLE labels (columns), use TabularPredictor for predicting one label (column)."
            )
        if (problem_types is not None) and (len(problem_types) != len(labels)):
            raise ValueError(
                "If provided, `problem_types` must have same length as `labels`"
            )
        if (eval_metrics is not None) and (len(eval_metrics) != len(labels)):
            raise ValueError(
                "If provided, `eval_metrics` must have same length as `labels`"
            )
        self.path = setup_outputdir(path, warn_if_exist=False)
        self.labels = labels
        self.consider_labels_correlation = consider_labels_correlation
        self.predictors = (
            {}
        )  # key = label, value = TabularPredictor or str path to the TabularPredictor for this label
        if eval_metrics is None:
            self.eval_metrics = {}
        else:
            self.eval_metrics = {labels[i]: eval_metrics[i] for i in range(len(labels))}
        problem_type = None
        eval_metric = None
        for i in range(len(labels)):
            label = labels[i]
            path_i = self.path + "Predictor_" + label
            if problem_types is not None:
                problem_type = problem_types[i]
            if eval_metrics is not None:
                eval_metric = eval_metrics[i]
            self.predictors[label] = MonotonePredictor(
                label=label,
                problem_type=problem_type,
                eval_metric=eval_metric,
                path=path_i,
                **kwargs,
            )


if __name__ == "__main__":
    params = get_ag_parameters().parse_args()
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
    ag_path = "monotonous_ag/" + ag_meta["ag_path"] + "/"

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
        print("not found, fitting")

        # create constraints for k1, k2 and k3. The appropriate format of the constraints
        # will be created on the fly to match the different model APIs.
        monotone_constraints = {"k1": -1, "k2": -1, "k3": -1}

        predictor = MonotoneMultilabelPredictor(
            path=ag_path,
            labels=objectives,
            problem_types=["regression"] * len(objectives),
            eval_metrics=[wmape] * len(objectives),
            consider_labels_correlation=False,
            monotone_constraints=monotone_constraints,
        )
        predictor.fit(
            train_data=train_data,
            num_stack_levels=1,
            num_bag_folds=4,
            hyperparameters={
                MonotoneTabularNeuralNetTorchModel: {},
                MonotoneXGBoostModel: {},
                MonotoneCatBoostModel: {},
                # "GBM": {},
                # "FASTAI": {},
            },
            # excluded_model_types=["KNN"],
            tuning_data=val_data,
            presets_lat=ag_sign.split(","),
            presets_io=None,
            use_bag_holdout=True,
            infer_limit=infer_limit,
            infer_limit_batch_size=infer_limit_batch_size,
            num_gpus=num_gpus,
            time_limit=time_limit,
        )

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
