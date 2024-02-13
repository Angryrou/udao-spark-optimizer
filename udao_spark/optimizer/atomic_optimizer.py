import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch as th
from udao.optimization.utils.moo_utils import is_pareto_efficient

from ..utils.constants import THETA_COMPILE
from ..utils.logging import logger
from .base_optimizer import BaseOptimizer


class AtomicOptimizer(BaseOptimizer):
    def extract_non_decision_df(self, non_decision_input: Dict) -> pd.DataFrame:
        """
        extract the non_decision dict to a DataFrame
        """
        df = pd.DataFrame.from_dict({0: non_decision_input}, orient="index")
        df.index.name = "id"
        df["id"] = df.index
        return df

    def solve(
        self,
        non_decision_input: Dict[str, Any],
        seed: Optional[int] = None,
        use_ag: bool = True,
        ag_model: Dict[str, str] = dict(),
        sample_mode: str = "random_sample",
        n_samples: int = 1,
        moo_mode: str = "BF",
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        t1 = time.perf_counter_ns()

        non_decision_df = self.extract_non_decision_df(non_decision_input)

        t2 = time.perf_counter_ns()
        if self.verbose:
            logger.info(f">> extracted non_decision_df in {(t2 - t1) / 1e6} ms")

        (
            graph_embeddings,
            non_decision_tabular_features,
        ) = self.extract_non_decision_embeddings_from_df(non_decision_df)

        t3 = time.perf_counter_ns()
        if self.verbose:
            logger.info(f">> extracted non_decision_embeddings in {(t3 - t2) / 1e6} ms")
            logger.debug("graph_embeddings shape: %s", graph_embeddings.shape)
            logger.warning(
                "non_decision_tabular_features is only used for "
                "MLP inference, shape: %s",
                non_decision_tabular_features.shape,
            )

        if sample_mode == "random":
            sampled_theta = self.sample_theta_all(
                n_samples=n_samples, seed=seed, normalize=not use_ag
            )[:, -len(self.decision_variables) :]
        elif sample_mode == "grid":
            raise NotImplementedError
        else:
            raise ValueError(f"sample_mode {sample_mode} is not supported")

        t4 = time.perf_counter_ns()
        if self.verbose:
            logger.info(f">> sampled theta in {(t4 - t3) / 1e6} ms")

        if use_ag:
            graph_embeddings = graph_embeddings.detach().cpu()
            objs_dict = self.get_objective_values_ag(
                graph_embeddings.tile(n_samples, 1).numpy(),
                pd.DataFrame(
                    np.tile(non_decision_df.values, (n_samples, 1)),
                    columns=non_decision_df.columns,
                ),
                sampled_theta,
                ag_model,
            )
        else:
            objs_dict = self.get_objective_values_mlp(
                graph_embeddings.tile(n_samples, 1),
                non_decision_tabular_features.tile(n_samples, 1),
                th.tensor(sampled_theta, dtype=self.dtype),
            )
        lat, cost = self.get_latencies_and_objectives(objs_dict)
        objs = np.vstack([lat, cost]).T.astype(np.float32)

        t5 = time.perf_counter_ns()
        if self.verbose:
            logger.info(f">> computed objective values in {(t5 - t4) / 1e6} ms")

        if moo_mode == "BF":
            po_mask = is_pareto_efficient(objs)
        else:
            raise ValueError(f"moo_mode {moo_mode} is not supported")
        po_objs = objs[po_mask]
        po_theta = sampled_theta[po_mask]

        t6 = time.perf_counter_ns()
        if self.verbose:
            logger.info(f">> filtered pareto optimal points in {(t6 - t5) / 1e6} ms")

        n_po = len(po_objs)
        if n_po == 0:
            return None, None

        logger.debug(f"po_objs: {po_objs}, po_theta: {po_theta}")
        non_decision_knobs = [
            v for v in THETA_COMPILE if v not in self.decision_variables
        ]
        n_vars = len(self.decision_variables)
        n_consts = len(non_decision_knobs)
        po_theta_full = np.hstack([np.zeros((n_po, n_consts)), po_theta])
        if use_ag:
            po_confs = self.sc.construct_configuration(po_theta_full)[:, -n_vars:]
        else:
            po_confs = self.sc.construct_configuration_from_norm(po_theta_full)[
                :, -n_vars:
            ]
        logger.debug(f"found {len(po_objs)} po points, po_confs: {po_confs}")

        t7 = time.perf_counter_ns()
        if self.verbose:
            logger.info(f">> constructed configurations in {(t7 - t6) / 1e6} ms")

        return po_objs, po_confs
