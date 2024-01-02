from itertools import chain
from logging import Logger
from typing import Any, Dict

import numpy as np

from udao_trace.parser.spark_parser import SparkParser
from udao_trace.utils import BenchmarkType, JsonHandler

from ...utils.constants import BETA, EPS
from ..utils import extract_compile_time_im


class InjectionExtractor:
    def __init__(
        self, benchmark_type: BenchmarkType, scale_factor: int, logger: Logger
    ) -> None:
        self.spark_parser = SparkParser(benchmark_type, scale_factor, logger)

    def get_q_compile(self, json_path: str) -> Dict[str, Any]:
        d = JsonHandler.load_json(json_path)["CompileTimeLQP"]
        size_in_mb = d["IM"]["inputSizeInBytes"] / 1024 / 1024
        row_count = d["IM"]["rowCount"]
        return {
            "lqp_str": JsonHandler.dump_to_string(
                self.spark_parser.drop_raw_plan(d["LQP"]), indent=None
            ),
            "IM-sizeInMB": size_in_mb,
            "IM-rowCount": row_count,
            "IM-sizeInMB-log": np.log(np.clip(size_in_mb, a_min=EPS, a_max=None)),
            "IM-rowCount-log": np.log(np.clip(row_count, a_min=EPS, a_max=None)),
        }

    def _get_one_qs_compile(self, qs: Dict[str, Any]) -> Dict[str, Any]:
        qs_dict = qs["QSLogical"]
        qs_lqp = JsonHandler.dump_to_string(
            self.spark_parser.drop_raw_plan(qs_dict), indent=None
        )
        size_in_mb, rows_count = extract_compile_time_im(qs_lqp)
        return {
            "qs_lqp": qs_lqp,
            "IM-sizeInMB-compile": size_in_mb,
            "IM-rowCount-compile": rows_count,
            "IM-sizeInMB-compile-log": np.log(
                np.clip(size_in_mb, a_min=EPS, a_max=None)
            ),
            "IM-rowCount-compile-log": np.log(
                np.clip(rows_count, a_min=EPS, a_max=None)
            ),
        }

    def _get_one_qs_runtime(self, qs: Dict[str, Any]) -> Dict[str, Any]:
        qs_dict = qs["QSLogical"]
        size_in_mb = qs["IM"]["inputSizeInBytes"] / 1024 / 1024
        row_count = qs["IM"]["inputRowCount"] * 1.0
        ss_dict = {}
        if "RunningQueryStageSnapshot" in qs:
            for k, v in qs["RunningQueryStageSnapshot"].items():
                if isinstance(v, list):
                    for tile, v_ in zip([0, 25, 50, 75, 100], v):
                        ss_dict[f"SS-{k}-{tile}tile"] = v_
                else:
                    ss_dict[f"SS-{k}"] = v
        else:
            raise ValueError("No RunningQueryStageSnapshot in qs")

        pd = np.array(list(chain.from_iterable(qs["PD"].values())))
        beta_ratios = [0.0, 0.0, 0.0]
        if pd.size > 0:
            mu, std, max_val, min_val = np.mean(pd), np.std(pd), np.max(pd), np.min(pd)
            beta_ratios = [std / mu, (max_val - mu) / mu, (max_val - min_val) / mu]
        beta_dict = {k: v for k, v in zip(BETA, beta_ratios)}

        return {
            "qs_lqp": JsonHandler.dump_to_string(
                self.spark_parser.drop_raw_plan(qs_dict), indent=None
            ),
            "IM-sizeInMB": size_in_mb,
            "IM-rowCount": row_count,
            "IM-sizeInMB-log": np.log(np.clip(size_in_mb, a_min=EPS, a_max=None)),
            "IM-rowCount-log": np.log(np.clip(row_count, a_min=EPS, a_max=None)),
            "PD": qs["PD"],
            **beta_dict,
            **ss_dict,
        }

    def get_qs_lqp(
        self, json_path: str, is_oracle: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        d = JsonHandler.load_json(json_path)["RuntimeQSs"]
        return {
            f"QS-{qs_id}": self._get_one_qs_runtime(qs)
            if is_oracle
            else self._get_one_qs_compile(qs)
            for qs_id, qs in d.items()
        }
