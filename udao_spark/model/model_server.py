from torchmetrics import WeightedMeanAbsolutePercentageError
from udao.model import UdaoModel, UdaoModule
from udao.model.utils.losses import WMAPELoss
from udao.optimization.utils.moo_utils import get_default_device
from udao.utils.logging import logger

from udao_trace.utils import JsonHandler

from .utils import (
    GraphAverageMLPParams,
    GraphTransformerMLPParams,
    get_graph_avg_mlp,
    get_graph_gtn_mlp,
)


class ModelServer:
    @classmethod
    def from_ckp_path(
        cls, model_sign: str, model_params_path: str, weights_path: str
    ) -> "ModelServer":
        if model_sign == "graph_avg":
            graph_avg_ml_params = GraphAverageMLPParams.from_dict(
                JsonHandler.load_json(model_params_path)
            )
            objectives = graph_avg_ml_params.iterator_shape.output_names
            model = get_graph_avg_mlp(graph_avg_ml_params)
            logger.info("MODEL DETAILS:\n")
            logger.info(model)
        elif model_sign == "graph_gtn":
            graph_gtn_ml_params = GraphTransformerMLPParams.from_dict(
                JsonHandler.load_json(model_params_path)
            )
            objectives = graph_gtn_ml_params.iterator_shape.output_names
            model = get_graph_gtn_mlp(graph_gtn_ml_params)
            logger.info("MODEL DETAILS:\n")
            logger.info(model)
        else:
            raise ValueError(f"Unknown model sign: {model_sign}")

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
        self.model.eval()
        self.objectives = module.objectives
        logger.info(f"Model loaded with objectives: {self.objectives}")
