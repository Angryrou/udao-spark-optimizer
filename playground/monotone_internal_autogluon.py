from __future__ import annotations

import logging
import time
from typing import List

from autogluon.common.utils.log_utils import set_logger_verbosity
from autogluon.common.utils.system_info import get_ag_system_info
from autogluon.common.utils.utils import setup_outputdir
from autogluon.core.constants import AUTO_WEIGHT, BALANCE_WEIGHT
from autogluon.core.learner import AbstractLearner
from autogluon.core.metrics import Scorer
from autogluon.core.models import BaggedEnsembleModel
from autogluon.core.trainer import AbstractTrainer
from autogluon.core.utils import generate_train_test_split
from autogluon.tabular import TabularPredictor
from autogluon.tabular.learner import DefaultLearner
from autogluon.tabular.trainer import AutoTrainer
from pandas import DataFrame

logger = logging.getLogger(__name__)


class MonotonePredictor(TabularPredictor):
    def __init__(self, label: str, monotone_constraints: dict[str, int], problem_type: str = None,
                 eval_metric: str | Scorer = None, path: str = None, verbosity: int = 2, log_to_file: bool = False,
                 log_file_path: str = "auto", sample_weight: str = None, weight_evaluation: bool = False,
                 groups: str = None, **kwargs):
        # if this fails, I can delete the super call and copy and paste the content of the method.
        super().__init__(label, problem_type, eval_metric, path, verbosity, log_to_file, log_file_path, sample_weight,
                         weight_evaluation, groups, **kwargs)

        self.monotone_constraints = monotone_constraints

        learner_type = kwargs.pop("learner_type", MonotoneDefaultLearner)
        learner_kwargs = kwargs.pop("learner_kwargs", dict())
        quantile_levels = kwargs.get("quantile_levels", None)

        self._learner: AbstractLearner = learner_type(
            path_context=path,
            label=label,
            feature_generator=None,
            eval_metric=eval_metric,
            problem_type=problem_type,
            quantile_levels=quantile_levels,
            sample_weight=self.sample_weight,
            weight_evaluation=self.weight_evaluation,
            groups=groups,
            **learner_kwargs,
        )
        self._learner_type = type(self._learner)
        self._trainer: MonotoneAutoTrainer = None

    def _fit(self, ag_fit_kwargs: dict, ag_post_fit_kwargs: dict):
        ag_fit_kwargs["monotone_constraints"] = self.monotone_constraints
        super()._fit(ag_fit_kwargs, ag_post_fit_kwargs)


class MonotoneAutoTrainer(AutoTrainer):
    def fit(
            self,
            X,
            y,
            hyperparameters,
            X_val=None,
            y_val=None,
            X_unlabeled=None,
            holdout_frac=0.1,
            num_stack_levels=0,
            core_kwargs: dict = None,
            aux_kwargs: dict = None,
            time_limit=None,
            infer_limit=None,
            infer_limit_batch_size=None,
            use_bag_holdout=False,
            groups=None,
            **kwargs,
    ):
        monotone_constraints = kwargs.pop("monotone_constraints", {})
        for key in kwargs:
            logger.warning(f"Warning: Unknown argument passed to `AutoTrainer.fit()`. Argument: {key}")

        if use_bag_holdout:
            if self.bagged_mode:
                logger.log(20,
                           f"use_bag_holdout={use_bag_holdout}, will use tuning_data as holdout (will not be used for early stopping).")
            else:
                logger.warning(
                    f"Warning: use_bag_holdout={use_bag_holdout}, but bagged mode is not enabled. use_bag_holdout will be ignored.")

        if (y_val is None) or (X_val is None):
            if not self.bagged_mode or use_bag_holdout:
                if groups is not None:
                    raise AssertionError(
                        f"Validation data must be manually specified if use_bag_holdout and groups are both specified.")
                if self.bagged_mode:
                    # Need at least 2 samples of each class in train data after split for downstream k-fold splits
                    # to ensure each k-fold has at least 1 sample of each class in training data
                    min_cls_count_train = 2
                else:
                    min_cls_count_train = 1
                X, X_val, y, y_val = generate_train_test_split(
                    X,
                    y,
                    problem_type=self.problem_type,
                    test_size=holdout_frac,
                    random_state=self.random_state,
                    min_cls_count_train=min_cls_count_train,
                )
                logger.log(
                    20,
                    f"Automatically generating train/validation split with holdout_frac={holdout_frac}, Train Rows: {len(X)}, Val Rows: {len(X_val)}"
                )
        elif self.bagged_mode:
            if not use_bag_holdout:
                # TODO: User could be intending to blend instead. Add support for blend stacking.
                #  This error message is necessary because when calculating out-of-fold predictions for user, we want to return them in the form given in train_data,
                #  but if we merge train and val here, it becomes very confusing from a users perspective, especially because we reset index, making it impossible to match
                #  the original train_data to the out-of-fold predictions from `predictor.predict_proba_oof()`.
                raise AssertionError(
                    "X_val, y_val is not None, but bagged mode was specified. "
                    "If calling from `TabularPredictor.fit()`, `tuning_data` should be None.\n"
                    "Default bagged mode does not use tuning data / validation data. "
                    "Instead, all data (`train_data` and `tuning_data`) should be combined and specified as `train_data`.\n"
                    "To avoid this error and use `tuning_data` as holdout data in bagged mode, "
                    "specify the following:\n"
                    "\tpredictor.fit(..., tuning_data=tuning_data, use_bag_holdout=True)"
                )

        # Log the hyperparameters dictionary so it easy to edit if the user wants.
        n_configs = sum([len(hyperparameters[k]) for k in hyperparameters.keys()])
        extra_log_str = ""
        display_all = (n_configs < 20) or (self.verbosity >= 3)
        if not display_all:
            extra_log_str = (
                f"Large model count detected ({n_configs} configs) ... " f"Only displaying the first 3 models of each family. To see all, set `verbosity=3`.\n"
            )
        log_str = f"{extra_log_str}User-specified model hyperparameters to be fit:\n" "{\n"
        if display_all:
            for k in hyperparameters.keys():
                log_str += f"\t'{k}': {hyperparameters[k]},\n"
        else:
            for k in hyperparameters.keys():
                log_str += f"\t'{k}': {hyperparameters[k][:3]},\n"
        log_str += "}"
        logger.log(20, log_str)

        self._train_multi_and_ensemble(
            X=X,
            y=y,
            X_val=X_val,
            y_val=y_val,
            X_unlabeled=X_unlabeled,
            hyperparameters=hyperparameters,
            num_stack_levels=num_stack_levels,
            time_limit=time_limit,
            core_kwargs=core_kwargs,
            aux_kwargs=aux_kwargs,
            infer_limit=infer_limit,
            infer_limit_batch_size=infer_limit_batch_size,
            groups=groups,
            monotone_constraints=monotone_constraints,
        )

    def train_multi_levels(
            self,
            X,
            y,
            hyperparameters: dict,
            X_val=None,
            y_val=None,
            X_unlabeled=None,
            base_model_names: List[str] = None,
            core_kwargs: dict = None,
            aux_kwargs: dict = None,
            level_start=1,
            level_end=1,
            time_limit=None,
            name_suffix: str = None,
            relative_stack=True,
            level_time_modifier=0.333,
            infer_limit=None,
            infer_limit_batch_size=None,
            monotone_constraints=None,
    ) -> List[str]:
        """
        Trains a multi-layer stack ensemble using the input data on the hyperparameters dict input.
            hyperparameters is used to determine the models used in each stack layer.
        If continuing a stack ensemble with level_start>1, ensure that base_model_names is set to the appropriate base models that will be used by the level_start level models.
        Trains both core and aux models.
            core models are standard models which are fit on the data features. Core models will also use model predictions if base_model_names was specified or if level != 1.
            aux models are ensemble models which only use the predictions of core models as features. These models never use the original features.

        level_time_modifier : float, default 0.333
            The amount of extra time given relatively to early stack levels compared to later stack levels.
            If 0, then all stack levels are given 100%/L of the time, where L is the number of stack levels.
            If 1, then all stack levels are given 100% of the time, meaning if the first level uses all of the time given to it, the other levels won't train.
            Time given to a level = remaining_time / remaining_levels * (1 + level_time_modifier), capped by total remaining time.

        Returns a list of the model names that were trained from this method call, in order of fit.
        """
        self._time_limit = time_limit
        self._time_train_start = time.time()
        time_train_start = self._time_train_start

        hyperparameters = self._process_hyperparameters(hyperparameters=hyperparameters)

        if relative_stack:
            if level_start != 1:
                raise AssertionError(f"level_start must be 1 when `relative_stack=True`. (level_start = {level_start})")
            level_add = 0
            if base_model_names:
                max_base_model_level = self.get_max_level(models=base_model_names)
                level_start = max_base_model_level + 1
                level_add = level_start - 1
                level_end += level_add
            if level_start != 1:
                hyperparameters_relative = {}
                for key in hyperparameters:
                    if isinstance(key, int):
                        hyperparameters_relative[key + level_add] = hyperparameters[key]
                    else:
                        hyperparameters_relative[key] = hyperparameters[key]
                hyperparameters = hyperparameters_relative

        core_kwargs = {} if core_kwargs is None else core_kwargs.copy()
        aux_kwargs = {} if aux_kwargs is None else aux_kwargs.copy()

        model_names_fit = []
        if level_start != level_end:
            logger.log(20,
                       f"AutoGluon will fit {level_end - level_start + 1} stack levels (L{level_start} to L{level_end}) ...")
        for level in range(level_start, level_end + 1):
            core_kwargs_level = core_kwargs.copy()
            aux_kwargs_level = aux_kwargs.copy()
            full_weighted_ensemble = aux_kwargs_level.pop("fit_full_last_level_weighted_ensemble", True) and (
                    level == level_end) and (level > 1)
            additional_full_weighted_ensemble = aux_kwargs_level.pop("full_weighted_ensemble_additionally",
                                                                     False) and full_weighted_ensemble
            if time_limit is not None:
                time_train_level_start = time.time()
                levels_left = level_end - level + 1
                time_left = time_limit - (time_train_level_start - time_train_start)
                time_limit_for_level = min(time_left / levels_left * (1 + level_time_modifier), time_left)
                time_limit_core = time_limit_for_level
                time_limit_aux = max(time_limit_for_level * 0.1, min(time_limit,
                                                                     360))  # Allows aux to go over time_limit, but only by a small amount
                core_kwargs_level["time_limit"] = core_kwargs_level.get("time_limit", time_limit_core)
                aux_kwargs_level["time_limit"] = aux_kwargs_level.get("time_limit", time_limit_aux)
            base_model_names, aux_models = self.stack_new_level(
                X=X,
                y=y,
                X_val=X_val,
                y_val=y_val,
                X_unlabeled=X_unlabeled,
                models=hyperparameters,
                level=level,
                base_model_names=base_model_names,
                core_kwargs=core_kwargs_level,
                aux_kwargs=aux_kwargs_level,
                name_suffix=name_suffix,
                infer_limit=infer_limit,
                infer_limit_batch_size=infer_limit_batch_size,
                full_weighted_ensemble=full_weighted_ensemble,
                additional_full_weighted_ensemble=additional_full_weighted_ensemble,
                monotone_constraints=monotone_constraints,
            )
            model_names_fit += base_model_names + aux_models
        if self.model_best is None and len(model_names_fit) != 0:
            self.model_best = self.get_model_best(can_infer=True, infer_limit=infer_limit, infer_limit_as_child=True)
        self._time_limit = None
        self.save()
        return model_names_fit

    def stack_new_level(
            self,
            X,
            y,
            models: Union[List[AbstractModel], dict],
            X_val=None,
            y_val=None,
            X_unlabeled=None,
            level=1,
            base_model_names: List[str] = None,
            core_kwargs: dict = None,
            aux_kwargs: dict = None,
            name_suffix: str = None,
            infer_limit=None,
            infer_limit_batch_size=None,
            full_weighted_ensemble: bool = False,
            additional_full_weighted_ensemble: bool = False,
            monotone_constraints=None
    ) -> (List[str], List[str]):
        """
        Similar to calling self.stack_new_level_core, except auxiliary models will also be trained via a call to self.stack_new_level_aux, with the models trained from self.stack_new_level_core used as base models.
        """
        if base_model_names is None:
            base_model_names = []
        core_kwargs = {} if core_kwargs is None else core_kwargs.copy()
        aux_kwargs = {} if aux_kwargs is None else aux_kwargs.copy()
        if level < 1:
            raise AssertionError(f"Stack level must be >= 1, but level={level}.")
        if base_model_names and level == 1:
            raise AssertionError(
                f"Stack level 1 models cannot have base models, but base_model_names={base_model_names}.")
        if name_suffix:
            core_kwargs["name_suffix"] = core_kwargs.get("name_suffix", "") + name_suffix
            aux_kwargs["name_suffix"] = aux_kwargs.get("name_suffix", "") + name_suffix
        core_models = self.stack_new_level_core(
            X=X,
            y=y,
            X_val=X_val,
            y_val=y_val,
            X_unlabeled=X_unlabeled,
            models=models,
            level=level,
            infer_limit=infer_limit,
            infer_limit_batch_size=infer_limit_batch_size,
            base_model_names=base_model_names,
            monotone_constraints=monotone_constraints,
            **core_kwargs,
        )

        aux_models = []
        if full_weighted_ensemble:
            full_aux_kwargs = aux_kwargs.copy()
            if additional_full_weighted_ensemble:
                full_aux_kwargs["name_extra"] = "_ALL"
            all_base_model_names = self.get_model_names(
                stack_name="core")  # Fit weighted ensemble on all previously fitted core models
            aux_models += self._stack_new_level_aux(X_val, y_val, X, y, all_base_model_names, level, infer_limit,
                                                    infer_limit_batch_size, **full_aux_kwargs)

        if (not full_weighted_ensemble) or additional_full_weighted_ensemble:
            aux_models += self._stack_new_level_aux(X_val, y_val, X, y, core_models, level, infer_limit,
                                                    infer_limit_batch_size, **aux_kwargs)

        return core_models, aux_models

    def _train_single_full(
            self,
            X,
            y,
            model: AbstractModel,
            X_unlabeled=None,
            X_val=None,
            y_val=None,
            X_pseudo=None,
            y_pseudo=None,
            feature_prune=False,
            hyperparameter_tune_kwargs=None,
            stack_name="core",
            k_fold=None,
            k_fold_start=0,
            k_fold_end=None,
            n_repeats=None,
            n_repeat_start=0,
            level=1,
            time_limit=None,
            fit_kwargs=None,
            compute_score=True,
            total_resources=None,
            **kwargs,
    ) -> List[str]:
        """
        Trains a model, with the potential to train multiple versions of this model with hyperparameter tuning and feature pruning.
        Returns a list of successfully trained and saved model names.
        Models trained from this method will be accessible in this Trainer.
        """
        monotone_constraints = kwargs.pop("monotone_constraints", {})
        model_fit_kwargs = self._get_model_fit_kwargs(
            X=X, X_val=X_val, time_limit=time_limit, k_fold=k_fold, fit_kwargs=fit_kwargs,
            ens_sample_weight=kwargs.get("ens_sample_weight", None)
        )
        if hyperparameter_tune_kwargs:
            if n_repeat_start != 0:
                raise ValueError(f"n_repeat_start must be 0 to hyperparameter_tune, value = {n_repeat_start}")
            elif k_fold_start != 0:
                raise ValueError(f"k_fold_start must be 0 to hyperparameter_tune, value = {k_fold_start}")
            # hpo_models (dict): keys = model_names, values = model_paths
            fit_log_message = f"Hyperparameter tuning model: {model.name} ..."
            if time_limit is not None:
                if time_limit <= 0:
                    logger.log(15, f"Skipping {model.name} due to lack of time remaining.")
                    return []
                fit_start_time = time.time()
                if self._time_limit is not None and self._time_train_start is not None:
                    time_left_total = self._time_limit - (fit_start_time - self._time_train_start)
                else:
                    time_left_total = time_limit
                fit_log_message += f" Tuning model for up to {round(time_limit, 2)}s of the {round(time_left_total, 2)}s of remaining time."
            logger.log(20, fit_log_message)
            try:
                if isinstance(model, BaggedEnsembleModel):
                    bagged_model_fit_kwargs = self._get_bagged_model_fit_kwargs(
                        k_fold=k_fold, k_fold_start=k_fold_start, k_fold_end=k_fold_end, n_repeats=n_repeats,
                        n_repeat_start=n_repeat_start
                    )
                    model_fit_kwargs.update(bagged_model_fit_kwargs)
                    hpo_models, hpo_results = model.hyperparameter_tune(
                        X=X,
                        y=y,
                        model=model,
                        X_val=X_val,
                        y_val=y_val,
                        X_unlabeled=X_unlabeled,
                        stack_name=stack_name,
                        level=level,
                        compute_score=compute_score,
                        hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
                        total_resources=total_resources,
                        **model_fit_kwargs,
                    )
                else:
                    hpo_models, hpo_results = model.hyperparameter_tune(
                        X=X,
                        y=y,
                        X_val=X_val,
                        y_val=y_val,
                        hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
                        total_resources=total_resources,
                        **model_fit_kwargs,
                    )
                if len(hpo_models) == 0:
                    logger.warning(
                        f"No model was trained during hyperparameter tuning {model.name}... Skipping this model.")
            except Exception as err:
                logger.exception(
                    f"Warning: Exception caused {model.name} to fail during hyperparameter tuning... Skipping this model.")
                logger.warning(err)
                del model
                model_names_trained = []
            else:
                # Commented out because it takes too much space (>>5 GB if run for an hour on a small-medium sized dataset)
                # self.hpo_results[model.name] = hpo_results
                model_names_trained = []
                self._extra_banned_names.add(model.name)
                for model_hpo_name, model_info in hpo_models.items():
                    model_hpo = self.load_model(model_hpo_name, path=os.path.relpath(model_info["path"], self.path),
                                                model_type=type(model))
                    logger.log(20, f"Fitted model: {model_hpo.name} ...")
                    if self._add_model(model=model_hpo, stack_name=stack_name, level=level):
                        model_names_trained.append(model_hpo.name)
        else:
            model_fit_kwargs.update(dict(X_pseudo=X_pseudo, y_pseudo=y_pseudo))
            if isinstance(model, BaggedEnsembleModel):
                bagged_model_fit_kwargs = self._get_bagged_model_fit_kwargs(
                    k_fold=k_fold, k_fold_start=k_fold_start, k_fold_end=k_fold_end, n_repeats=n_repeats,
                    n_repeat_start=n_repeat_start
                )
                model_fit_kwargs.update(bagged_model_fit_kwargs)
            model_names_trained = self._train_and_save(
                X=X,
                y=y,
                model=model,
                X_val=X_val,
                y_val=y_val,
                X_unlabeled=X_unlabeled,
                stack_name=stack_name,
                level=level,
                compute_score=compute_score,
                total_resources=total_resources,
                monotone_constraints=monotone_constraints,
                **model_fit_kwargs,
            )
        self.save()
        return model_names_trained


class MonotoneDefaultLearner(DefaultLearner):
    def __init__(self, trainer_type=MonotoneAutoTrainer, **kwargs):
        super().__init__(trainer_type, **kwargs)

    def _fit(
            self,
            X: DataFrame,
            X_val: DataFrame = None,
            X_unlabeled: DataFrame = None,
            holdout_frac=0.1,
            num_bag_folds=0,
            num_bag_sets=1,
            time_limit=None,
            infer_limit=None,
            infer_limit_batch_size=None,
            verbosity=2,
            **trainer_fit_kwargs,
    ):
        """Arguments:
        X (DataFrame): training data
        X_val (DataFrame): data used for hyperparameter tuning. Note: final model may be trained using this data as well as training data
        X_unlabeled (DataFrame): data used for pretraining a model. This is same data format as X, without label-column. This data is used for semi-supervised learning.
        holdout_frac (float): Fraction of data to hold out for evaluating validation performance (ignored if X_val != None, ignored if kfolds != 0)
        num_bag_folds (int): kfolds used for bagging of models, roughly increases model training time by a factor of k (0: disabled)
        num_bag_sets (int): number of repeats of kfold bagging to perform (values must be >= 1),
            total number of models trained during bagging = num_bag_folds * num_bag_sets
        """
        self._time_limit = time_limit
        if time_limit:
            logger.log(20, f"Beginning AutoGluon training ... Time limit = {time_limit}s")
        else:
            logger.log(20, "Beginning AutoGluon training ...")
        logger.log(20, f'AutoGluon will save models to "{self.path}"')
        include_gpu_count = False
        if verbosity >= 3:
            include_gpu_count = True
        msg = get_ag_system_info(path=self.path, include_gpu_count=include_gpu_count)
        logger.log(20, msg)
        logger.log(20, f"Train Data Rows:    {len(X)}")
        logger.log(20, f"Train Data Columns: {len([column for column in X.columns if column != self.label])}")
        if X_val is not None:
            logger.log(20, f"Tuning Data Rows:    {len(X_val)}")
            logger.log(20, f"Tuning Data Columns: {len([column for column in X_val.columns if column != self.label])}")
        logger.log(20, f"Label Column:       {self.label}")
        time_preprocessing_start = time.time()
        self._pre_X_rows = len(X)
        if self.problem_type is None:
            self.problem_type = self.infer_problem_type(y=X[self.label])
        logger.log(20, f"Problem Type:       {self.problem_type}")
        if self.groups is not None:
            num_bag_sets = 1
            num_bag_folds = len(X[self.groups].unique())
        X_og = None if infer_limit_batch_size is None else X
        logger.log(20, "Preprocessing data ...")
        X, y, X_val, y_val, X_unlabeled, holdout_frac, num_bag_folds, groups = self.general_data_processing(X, X_val,
                                                                                                            X_unlabeled,
                                                                                                            holdout_frac,
                                                                                                            num_bag_folds)
        if X_og is not None:
            infer_limit = self._update_infer_limit(X=X_og, infer_limit_batch_size=infer_limit_batch_size,
                                                   infer_limit=infer_limit)

        self._post_X_rows = len(X)
        time_preprocessing_end = time.time()
        self._time_fit_preprocessing = time_preprocessing_end - time_preprocessing_start
        logger.log(20,
                   f"Data preprocessing and feature engineering runtime = {round(self._time_fit_preprocessing, 2)}s ...")
        if time_limit:
            time_limit_trainer = time_limit - self._time_fit_preprocessing
        else:
            time_limit_trainer = None

        trainer = self.trainer_type(
            path=self.model_context,
            problem_type=self.label_cleaner.problem_type_transform,
            eval_metric=self.eval_metric,
            num_classes=self.label_cleaner.num_classes,
            quantile_levels=self.quantile_levels,
            feature_metadata=self.feature_generator.feature_metadata,
            low_memory=True,
            k_fold=num_bag_folds,  # TODO: Consider moving to fit call
            n_repeats=num_bag_sets,  # TODO: Consider moving to fit call
            sample_weight=self.sample_weight,
            weight_evaluation=self.weight_evaluation,
            save_data=self.cache_data,
            random_state=self.random_state,
            verbosity=verbosity,
        )

        self.trainer_path = trainer.path
        if self.eval_metric is None:
            self.eval_metric = trainer.eval_metric

        self.save()
        trainer.fit(
            X=X,
            y=y,
            X_val=X_val,
            y_val=y_val,
            X_unlabeled=X_unlabeled,
            holdout_frac=holdout_frac,
            time_limit=time_limit_trainer,
            infer_limit=infer_limit,
            infer_limit_batch_size=infer_limit_batch_size,
            groups=groups,
            **trainer_fit_kwargs,
        )
        self.save_trainer(trainer=trainer)
        time_end = time.time()
        self._time_fit_training = time_end - time_preprocessing_end
        self._time_fit_total = time_end - time_preprocessing_start
        logger.log(20,
                   f'AutoGluon training complete, total runtime = {round(self._time_fit_total, 2)}s ... Best model: "{trainer.model_best}"')
