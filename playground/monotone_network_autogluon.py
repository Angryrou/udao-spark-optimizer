from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Tuple, Any

from autogluon.core.constants import QUANTILE, REGRESSION, SOFTCLASS, BINARY, MULTICLASS
from autogluon.tabular.models import TabularNeuralNetTorchModel, XGBoostModel, CatBoostModel
import os
import torch
import torch.nn as nn
import numpy as np
import logging

from autogluon.tabular.models.tabular_nn.torch.torch_network_modules import EmbedNet
from autogluon.tabular.models.tabular_nn.utils.nn_architecture_utils import get_embed_sizes
from numpy._typing import ArrayLike

logger = logging.getLogger(__name__)


class MonotoneXGBoostModel(XGBoostModel):

    def _fit(self, X, y, X_val=None, y_val=None, time_limit=None, num_gpus=0, num_cpus=None, sample_weight=None,
             sample_weight_val=None, verbosity=2, **kwargs):
        monotone_constraints = kwargs.get("monotone_constraints", {})
        monotone_constraints_list = [0 for _ in range(len(X.columns))]
        for idx, col in enumerate(X.columns):
            if col in monotone_constraints.keys():
                monotone_constraints_list[idx] = monotone_constraints[col]
        monotone_constraints_str = str(tuple(monotone_constraints_list))
        self.params.update({"monotone_constraints": monotone_constraints_str})
        super()._fit(X, y, X_val, y_val, time_limit, num_gpus, num_cpus, sample_weight, sample_weight_val, verbosity,
                     **kwargs)


class MonotoneCatBoostModel(CatBoostModel):
    def _fit(self, X, y, X_val=None, y_val=None, time_limit=None, num_gpus=0, num_cpus=-1, sample_weight=None,
             sample_weight_val=None, **kwargs):
        monotone_constraints = kwargs.get("monotone_constraints", {})
        monotone_constraints_list = [0 for _ in range(len(X.columns))]
        for idx, col in enumerate(X.columns):
            if col in monotone_constraints.keys():
                monotone_constraints_list[idx] = monotone_constraints[col]
        monotone_constraints_str = "({})".format(",".join([str(val) for val in monotone_constraints_list]))
        self.params.update({"monotone_constraints": monotone_constraints_str})
        return super()._fit(X, y, X_val, y_val, time_limit, num_gpus, num_cpus, sample_weight, sample_weight_val,
                            **kwargs)


class MonotoneTabularNeuralNetTorchModel(TabularNeuralNetTorchModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # noinspection SpellCheckingInspection
    def _get_net(self, train_dataset, params):
        # TODO(glachaud): rewrite the function to add my call my monotone network

        # set network params
        params = self._set_net_defaults(train_dataset, params)
        self.model = MonotoneEmbedNet(
            problem_type=self.problem_type,
            monotone_constraints=self.monotone_constraints,
            num_net_outputs=self._get_num_net_outputs(),
            quantile_levels=self.quantile_levels,
            train_dataset=train_dataset,
            device=self.device,
            **params,
        )
        self.model = self.model.to(self.device)
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def fit(self, **kwargs):
        self.monotone_constraints = kwargs.get('monotone_constraints', {})
        return super().fit(**kwargs)

    def _fit(self, X, y, X_val=None, y_val=None, time_limit=None, sample_weight=None, num_cpus=1, num_gpus=0,
             reporter=None, verbosity=2, **kwargs):
        monotone_constraints = [0 for _ in range(len(X.columns))]
        for idx, col in enumerate(X.columns):
            if col in self.monotone_constraints.keys():
                monotone_constraints[idx] = self.monotone_constraints[col]
        self.monotone_constraints = monotone_constraints
        return super()._fit(X, y, X_val, y_val, time_limit, sample_weight, num_cpus, num_gpus, reporter, verbosity,
                            **kwargs)


class MonotoneEmbedNet(EmbedNet):
    def __init__(self, problem_type,
                 monotone_constraints=None,
                 num_net_outputs=None,
                 quantile_levels=None,
                 train_dataset=None,
                 architecture_desc=None,
                 device=None, **kwargs):
        if (architecture_desc is None) and (train_dataset is None):
            raise ValueError("train_dataset cannot = None if architecture_desc=None")
        # we don't want to initialize the class with the EmbedNet constructor, so we use the grandparent class.
        nn.Module.__init__(self)
        self.problem_type = problem_type
        if self.problem_type == QUANTILE:
            self.register_buffer("quantile_levels", torch.Tensor(quantile_levels).float().reshape(1, -1))
        self.device = torch.device("cpu") if device is None else device
        if architecture_desc is None:
            if monotone_constraints is None:
                monotone_constraints = [0]
            params = self._set_params(**kwargs)
            # adaptively specify network architecture based on training dataset
            self.from_logits = False
            self.has_vector_features = train_dataset.has_vector_features
            self.has_embed_features = train_dataset.has_embed_features
            if self.has_embed_features:
                # noinspection SpellCheckingInspection
                num_categs_per_feature = train_dataset.getNumCategoriesEmbeddings()
                embed_dims = get_embed_sizes(train_dataset, params, num_categs_per_feature)
            if self.has_vector_features:
                vector_dims = train_dataset.data_list[train_dataset.vectordata_index].shape[-1]
        else:
            # ignore train_dataset, params, etc. Recreate architecture based on description:
            self.architecture_desc = architecture_desc
            self.has_vector_features = architecture_desc["has_vector_features"]
            self.has_embed_features = architecture_desc["has_embed_features"]
            self.from_logits = architecture_desc["from_logits"]
            params = architecture_desc["params"]
            if self.has_embed_features:
                num_categs_per_feature = architecture_desc["num_categs_per_feature"]
                embed_dims = architecture_desc["embed_dims"]
            if self.has_vector_features:
                vector_dims = architecture_desc["vector_dims"]
        # init input size
        input_size = 0

        # define embedding layer:
        if self.has_embed_features:
            self.embed_blocks = nn.ModuleList()
            # noinspection PyUnboundLocalVariable
            for i in range(len(num_categs_per_feature)):
                # noinspection PyUnboundLocalVariable
                self.embed_blocks.append(
                    nn.Embedding(num_embeddings=num_categs_per_feature[i], embedding_dim=embed_dims[i]))
                input_size += embed_dims[i]

        # update input size
        if self.has_vector_features:
            # noinspection PyUnboundLocalVariable
            input_size += vector_dims

        # activation
        act_fn = nn.Identity()
        if params["activation"] == "elu":
            act_fn = nn.ELU()
        elif params["activation"] == "relu":
            act_fn = nn.ReLU()
        elif params["activation"] == "tanh":
            act_fn = nn.Tanh()

        ######################### Code to change ######################################
        layers = []
        if params["use_batchnorm"]:
            layers.append(nn.BatchNorm1d(input_size))
        layers.append(MonoDenseTorch(input_size, params["hidden_size"],
                                     monotonicity_indicator=monotone_constraints,
                                     activation=act_fn))
        # layers.append(nn.Linear(input_size, params["hidden_size"]))
        # layers.append(act_fn)
        for _ in range(params["num_layers"] - 1):
            if params["use_batchnorm"]:
                layers.append(nn.BatchNorm1d(params["hidden_size"]))
            layers.append(nn.Dropout(params["dropout_prob"]))
            layers.append(MonoDenseTorch(params["hidden_size"], params["hidden_size"],
                                         monotonicity_indicator=1,
                                         activation=act_fn))
            # layers.append(nn.Linear(params["hidden_size"], params["hidden_size"]))
            # layers.append(act_fn)
        layers.append(MonoDenseTorch(params["hidden_size"], num_net_outputs,
                                     monotonicity_indicator=1,
                                     activation=None))
        # layers.append(nn.Linear(params["hidden_size"], num_net_outputs))
        self.main_block = nn.Sequential(*layers)
        ######################### End Code to change ######################################

        if self.problem_type in [REGRESSION, QUANTILE]:  # set range for output
            y_range = params["y_range"]  # Used specifically for regression. = None for classification.
            self.y_constraint = None  # determines if Y-predictions should be constrained
            if y_range is not None:
                if y_range[0] == -np.inf and y_range[1] == np.inf:
                    self.y_constraint = None  # do not worry about Y-range in this case
                elif y_range[0] >= 0 and y_range[1] == np.inf:
                    self.y_constraint = "nonnegative"
                elif y_range[0] == -np.inf and y_range[1] <= 0:
                    self.y_constraint = "nonpositive"
                else:
                    self.y_constraint = "bounded"
                self.y_lower = y_range[0]
                self.y_upper = y_range[1]
                self.y_span = self.y_upper - self.y_lower

        if self.problem_type == QUANTILE:
            self.alpha = params["alpha"]  # for huber loss
        if self.problem_type == SOFTCLASS:
            self.log_softmax = torch.nn.LogSoftmax(dim=1)
        if self.problem_type in [BINARY, MULTICLASS, SOFTCLASS]:
            self.softmax = torch.nn.Softmax(dim=1)
        if architecture_desc is None:  # Save Architecture description
            self.architecture_desc = {
                "has_vector_features": self.has_vector_features,
                "has_embed_features": self.has_embed_features,
                "params": params,
                "num_net_outputs": num_net_outputs,
                "from_logits": self.from_logits,
            }
            if self.has_embed_features:
                self.architecture_desc["num_categs_per_feature"] = num_categs_per_feature
                self.architecture_desc["embed_dims"] = embed_dims
            if self.has_vector_features:
                self.architecture_desc["vector_dims"] = vector_dims


def get_saturated_activation(
        convex_activation: Callable[[torch.Tensor], torch.Tensor],
        concave_activation: Callable[[torch.Tensor], torch.Tensor],
        a: float = 1.0,
        c: float = 1.0,
) -> Callable[[torch.Tensor], torch.Tensor]:
    def saturated_activation(
            x: torch.Tensor,
    ) -> torch.Tensor:
        cc: torch.Tensor = convex_activation(torch.ones_like(x) * c)
        # Pytorch seems to handle typing inappropriately.
        return a * torch.where(
            torch.lt(x, 0),
            convex_activation(x + c) - cc,
            concave_activation(x - c) + cc,
        )

    return saturated_activation  # type: ignore


class ConvexActivation:
    """
    Applies a convex activation function element wise.
    """

    def __init__(self, activation: Callable[[torch.Tensor], torch]):
        self.activation = activation

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(inputs)


class ConcaveActivation:
    """
    Applies a concave activation function element wise. The function
    is defined with respect to a given convex activation function.
    """

    def __init__(self, activation: Callable[[torch.Tensor], torch]):
        self.activation = activation

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        return -self.activation(-inputs)


class SaturatedActivation:
    """
    Applies a saturated activation function element wise. The function
    is defined with respect to a given convex activation function.
    """

    def __init__(self, activation: Callable[[torch.Tensor], torch],
                 a: float = 1.0,
                 c: float = 1.0):
        self.convex_activation = ConvexActivation(activation)
        self.concave_activation = ConcaveActivation(activation)
        self.a: float = a
        self.c: float = c

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        cc: torch.Tensor = self.convex_activation(torch.ones_like(inputs) * self.c)
        return self.a * torch.where(
            torch.lt(inputs, 0),
            self.convex_activation(inputs + self.c) - cc,
            self.concave_activation(inputs - self.c) + cc
        )


# @lru_cache
# def get_activation_functions(
#         activation: Callable[[torch.Tensor], torch.Tensor] = None
# ) -> Tuple[
#     Callable[[torch.Tensor], torch.Tensor],
#     Callable[[torch.Tensor], torch.Tensor],
#     Callable[[torch.Tensor], torch.Tensor],
# ]:
#     # TODO (glachaud): this is not serializable with pickle (closures are not detected)
#     # contrary to Keras, we don't support activation creation using strings
#     convex_activation = activation
#
#     def concave_activation(x: torch.Tensor) -> torch.Tensor:
#         return -convex_activation(-x)
#
#     saturated_activation = get_saturated_activation(
#         convex_activation, concave_activation
#     )
#     return convex_activation, concave_activation, saturated_activation


def apply_activations(
        x: torch.Tensor,
        *,
        units: int,
        convex_activation: Callable[[torch.Tensor], torch.Tensor],
        concave_activation: Callable[[torch.Tensor], torch.Tensor],
        saturated_activation: Callable[[torch.Tensor], torch.Tensor],
        is_convex: bool = False,
        is_concave: bool = False,
        activation_weights: Tuple[float, float, float] = (7.0, 7.0, 2.0),
) -> torch.Tensor:
    if convex_activation is None:
        return x

    elif is_convex:
        normalized_activation_weights = np.array([1.0, 0.0, 0.0])
    elif is_concave:
        normalized_activation_weights = np.array([0.0, 1.0, 0.0])
    else:
        if len(activation_weights) != 3:
            raise ValueError(f"activation_weights={activation_weights}")
        if (np.array(activation_weights) < 0).any():
            raise ValueError(f"activation_weights={activation_weights}")
        normalized_activation_weights = np.array(activation_weights) / sum(
            activation_weights
        )

    # noinspection PyTypeChecker
    s_convex: int = round(normalized_activation_weights[0] * units)
    # noinspection PyTypeChecker
    s_concave: int = round(normalized_activation_weights[1] * units)
    s_saturated: int = units - s_convex - s_concave

    x_convex, x_concave, x_saturated = torch.split(
        x, [s_convex, s_concave, s_saturated], dim=-1
    )

    y_convex = convex_activation(x_convex)
    y_concave = concave_activation(x_concave)
    y_saturated = saturated_activation(x_saturated)

    y = torch.concat([y_convex, y_concave, y_saturated], dim=-1)

    return y


def get_monotonicity_indicator(
        monotonicity_indicator: ArrayLike,
        *,
        in_features: int,
        out_features: int,
) -> torch.Tensor:
    # convert to tensor if needed and make it broadcastable to the kernel
    monotonicity_indicator = np.array(monotonicity_indicator)
    # I don't think this works
    # if len(monotonicity_indicator.shape) < 2:
    #     monotonicity_indicator = np.reshape(monotonicity_indicator, (-1, 1))
    if len(monotonicity_indicator.shape) > 2:
        raise ValueError(
            f"monotonicity_indicator has rank greater than 2: {monotonicity_indicator.shape}"
        )

    if not np.all(
            (monotonicity_indicator == -1)
            | (monotonicity_indicator == 0)
            | (monotonicity_indicator == 1)
    ):
        raise ValueError(
            f"Each element of monotonicity_indicator must be one of -1, 0, 1, but it is: '{monotonicity_indicator}'"
        )

    monotonicity_indicator_broadcast = np.broadcast_to(
        monotonicity_indicator, shape=(out_features, in_features)
    )

    return torch.tensor(monotonicity_indicator_broadcast)


def apply_monotonicity_indicator_to_kernel(
        kernel: torch.Tensor,
        monotonicity_indicator: torch.Tensor,
) -> torch.Tensor:
    # convert to tensor if needed and make it broadcastable to the kernel

    # Pytorch stores the weight matrix with the opposite convention from Keras.
    # absolute value of the kernel
    abs_kernel = torch.abs(kernel)

    # replace original kernel values for positive or negative ones where needed
    xs = torch.where(
        torch.eq(monotonicity_indicator, 1),
        abs_kernel,
        kernel,
    )
    xs = torch.where(torch.eq(monotonicity_indicator, -1), -abs_kernel, xs)

    # We return the input that has been transposed back into the original shape.
    return xs


@contextmanager
def replace_kernel_using_monotonicity_indicator(
        layer: torch.nn.Linear,
        monotonicity_indicator: torch.Tensor,
) -> Generator[None, None, None]:
    old_kernel = layer.weight.data

    layer.weight.data = apply_monotonicity_indicator_to_kernel(
        layer.weight.data, monotonicity_indicator
    )
    try:
        yield
    finally:
        pass
        # layer.weight.data = old_kernel


class MonoDenseTorch(torch.nn.Linear):
    """Monotonic counterpart of the regular Linear Layer of PyTorch

    This is an implementation of the Monotonic Dense Unit or Constrained Monotone Fully Connected Layer.

    - the parameter `monotonicity_indicator` corresponds to **t** in the figure below, and

    - parameters `is_convex`, `is_concave` and `activation_weights` are used to calculate
     the activation selector **s** as follows:

        - if `is_convex` or `is_concave` is **True**, then the activation selector **s**
        will be (`units`, 0, 0) and (0, `units`, 0), respectively.

        - if both  `is_convex` or `is_concave` is **False**, then the `activation_weights` represent ratios
         between $\\breve{s}$, $\\hat{s}$ and $\\tilde{s}$, respectively. E.g.
          if `activation_weights = (2, 2, 1)` and `units = 10`, then

    $$
    (\\breve{s}, \\hat{s}, \\tilde{s}) = (4, 4, 2)
    $$
    """

    def __init__(
            self,
            in_features: int,
            out_features: int,
            monotonicity_indicator,
            *,
            activation: Callable[[torch.Tensor], torch.Tensor] = None,
            is_convex: bool = False,
            is_concave: bool = False,
            activation_weights: Tuple[float, float, float] = (7.0, 7.0, 2.0),
            **kwargs: Any,
    ):
        """Constructs a new MonoDense instance.

        Params:
            in_features: Positive integer, dimensionality of the input space.
            out_features: Positive integer, dimensionality of the output space.
            activation: Activation function to use, it is assumed to be convex monotonically
                increasing function such as "relu" or "elu"
            monotonicity_indicator: Vector to indicate which of the inputs are monotonically increasing or
                monotonically decreasing or non-monotonic. Has value 1 for monotonically increasing,
                -1 for monotonically decreasing and 0 for non-monotonic.
            is_convex: convex if set to True
            is_concave: concave if set to True
            activation_weights: relative weights for each type of activation, the default is (1.0, 1.0, 1.0).
                Ignored if is_convex or is_concave is set to True
            **kwargs: passed as kwargs to the constructor of `Dense`

        Raise:
            ValueError:
                - if both **is_concave** and **is_convex** are set to **True**, or
                - if any component of activation_weights is negative or there is not exactly three components
        """
        self.monotonicity_indicator = monotonicity_indicator
        if is_convex and is_concave:
            raise ValueError(
                "The model cannot be set to be both convex and concave (only linear functions are both)."
            )

        if len(activation_weights) != 3:
            raise ValueError(
                f"There must be exactly three components of activation_weights"
                f", but we have this instead: {activation_weights}."
            )

        if (np.array(activation_weights) < 0).any():
            raise ValueError(
                f"Values of activation_weights must be non-negative, but we have this instead: {activation_weights}."
            )

        super().__init__(in_features=in_features, out_features=out_features, **kwargs)

        self.in_features = in_features
        self.out_features = out_features
        self.org_activation = activation
        self.activation_weights = activation_weights
        self.is_convex = is_convex
        self.is_concave = is_concave
        self.convex_activation = ConvexActivation(self.org_activation)
        self.concave_activation = ConcaveActivation(self.org_activation)
        self.saturated_activation = SaturatedActivation(self.org_activation)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Call

        Args:
            inputs: input tensor of shape (batch_size, ..., x_length)

        Returns:
            N-D tensor with shape: `(batch_size, ..., units)`.

        """
        # inputs, monotonicity_indicator = inputs
        monotonicity_indicator = get_monotonicity_indicator(
            monotonicity_indicator=self.monotonicity_indicator,
            in_features=self.in_features,
            out_features=self.out_features,
        )

        # calculate W'*x+y after we replace the kernel according to monotonicity vector
        with replace_kernel_using_monotonicity_indicator(
                self, monotonicity_indicator=monotonicity_indicator
        ):
            h = super().forward(inputs)

        if self.org_activation is None:
            return h

        y = apply_activations(
            h,
            units=self.out_features,
            convex_activation=self.convex_activation,
            concave_activation=self.concave_activation,
            saturated_activation=self.saturated_activation,
            is_convex=self.is_convex,
            is_concave=self.is_concave,
            activation_weights=self.activation_weights,
        )

        return y
