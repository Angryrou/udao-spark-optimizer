from torchmetrics import WeightedMeanAbsolutePercentageError
from udao.model import UdaoModel, UdaoModule
from udao.model.utils.losses import WMAPELoss
from udao.optimization.utils.moo_utils import get_default_device
from udao.utils.logging import logger

from udao_trace.utils import JsonHandler

from .utils import GraphAverageMLPParams, get_graph_avg_mlp


class ModelServer:
    @classmethod
    def from_ckp_path(
        cls, model_sign: str, model_params_path: str, weights_path: str
    ) -> "ModelServer":
        if model_sign == "graph_avg":
            model_params = GraphAverageMLPParams.from_dict(
                JsonHandler.load_json(model_params_path)
            )
            model = get_graph_avg_mlp(model_params)
            model_sign = "graph_avg"
        else:
            raise ValueError(f"Unknown model sign: {model_sign}")
        objectives = model_params.iterator_shape.output_names
        module = UdaoModule.load_from_checkpoint(
            weights_path,
            map_location=get_default_device(),
            model=model,
            objectives=objectives,
            loss=WMAPELoss(),
            metrics=[WeightedMeanAbsolutePercentageError],
        )
        return cls(model_sign, module)

    def __init__(self, model_sign: str, module: UdaoModule):
        self.model_sign = model_sign
        self.module = module
        if not isinstance(module.model, UdaoModel):
            raise TypeError(f"Unknown model type: {type(module.model)}")
        self.model: UdaoModel = module.model
        self.objectives = module.objectives
        logger.info(f"Model loaded with objectives: {self.objectives}")
