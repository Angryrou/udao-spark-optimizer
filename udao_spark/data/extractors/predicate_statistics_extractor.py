import re
from datetime import date, datetime
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from udao.data import StaticExtractor, TabularContainer


def get_pred_triplets(pred: str) -> str:
    if "CASE WHEN" in pred or "OR" in pred:
        return ""
    pattern = r"(\w+)(?:#\d+)\s*([<>=!]{1,2})\s*(\d{4}-\d{2}-\d{2}|\d+\.\d+|\d+)"
    matches = re.finditer(pattern, pred)  # type: ignore
    triplets = [match.groups() for match in matches]
    if any(["_" not in triplet[0] for triplet in triplets]):
        return ""
    triplets_str = ";".join([",".join(triplet) for triplet in triplets])
    return triplets_str


def date_format_match(val_str: str) -> bool:
    if re.match(r"^\d{4}-\d{2}-\d{2}$", val_str):
        return True
    else:
        return False


def days_since_epoch(date_str: str) -> int:
    """
    Calculate the number of days from January 1, 1970, to the given date.

    Args:
    date_str (str): Date in "yyyy-mm-dd" format.

    Returns:
    int: Number of days between 1970-01-01 and the given date.
    """
    epoch_start = datetime(1970, 1, 1)
    given_date = datetime.strptime(date_str, "%Y-%m-%d")

    # Calculate the difference in days
    delta = given_date - epoch_start
    return delta.days


def hist_hits(hist: np.ndarray, op: str, val: float) -> np.ndarray:
    res = np.zeros(len(hist) - 1)
    if op == "=" or op == "==":
        matches = np.where(hist == val)[0]
        if len(matches) == 0:
            return res
        res[matches[:-1]] = 1
        return res
    elif op == ">":
        matches = np.where(hist > val)[0]
        if len(matches) == 0:
            return res
        res[matches[:-1]] = 1
        if matches[0] > 0:
            res[matches[0] - 1] = (hist[matches[0]] - val) / (
                hist[matches[0]] - hist[matches[0] - 1]
            )
        return res
    elif op == ">=":
        matches = np.where(hist >= val)[0]
        if len(matches) == 0:
            return res
        res[matches[:-1]] = 1
        if matches[0] > 0:
            res[matches[0] - 1] = (hist[matches[0]] - val) / (
                hist[matches[0]] - hist[matches[0] - 1]
            )
        return res
    elif op == "<":
        matches = np.where(hist < val)[0]
        if len(matches) == 0:
            return res
        res[matches[:-1]] = 1
        if matches[-1] < len(hist) - 1:
            res[matches[-1]] = (val - hist[matches[-1]]) / (
                hist[matches[-1] + 1] - hist[matches[-1]]
            )
        return res
    elif op == "<=":
        matches = np.where(hist <= val)[0]
        if len(matches) == 0:
            return res
        res[matches[:-1]] = 1
        if matches[-1] < len(hist) - 1:
            res[matches[-1]] = (val - hist[matches[-1]]) / (
                hist[matches[-1] + 1] - hist[matches[-1]]
            )
        return res
    else:
        raise ValueError(f"Invalid operator: {op}")


class PredicateHistogramExtractor(StaticExtractor[TabularContainer]):
    def __init__(
        self,
        hists: Dict[str, np.ndarray],
        extract_operations: Callable[
            [pd.DataFrame, Callable], Tuple[Dict[int, List[int]], List[str]]
        ],
        max_num_of_column_predicates: int = 3,
    ):
        super().__init__()
        if len(hists) == 0:
            raise ValueError("No histograms provided.")
        self.hists = hists
        self.hists_bins = len(list(hists.values())[0]) - 1  # 50
        self.extract_operations = extract_operations
        self.max_triplets = max_num_of_column_predicates  # 3

    def hist_encoding(self, pred_triplets: List[str]) -> np.ndarray:
        n = self.hists_bins
        ress = np.zeros((len(pred_triplets), n * self.max_triplets))
        for i, operations in enumerate(pred_triplets):
            if operations == "":
                continue
            triplets = [operation.split(",") for operation in operations.split(";")]
            if any(triplet[0] not in self.hists for triplet in triplets):
                continue

            triplets_dict = {}
            for operation in operations.split(";"):
                col, op, val_str = operation.split(",")
                if date_format_match(val_str):
                    val = float(days_since_epoch(val_str))
                else:
                    try:
                        val = float(val_str)
                    except Exception as e:
                        raise ValueError(f"Error: {operations}, {e}")
                if col not in triplets_dict:
                    triplets_dict[col] = {op: val}
                else:
                    if op not in triplets_dict[col]:
                        triplets_dict[col][op] = val
                    else:
                        if op in ["<", "<="]:
                            triplets_dict[col][op] = min(triplets_dict[col][op], val)
                        elif op in [">", ">="]:
                            triplets_dict[col][op] = max(triplets_dict[col][op], val)
                        else:  # triplet[1] == "="
                            if triplets_dict[col][op] != val:
                                raise Exception(f"Error for {operations}")
                if len(triplets_dict[col]) > self.max_triplets:
                    raise Exception(
                        f"{self.max_triplets} is too small for {operations}"
                    )

            for j, (col, op_vals) in enumerate(triplets_dict.items()):
                res_col = np.ones(n)
                for op, val in op_vals.items():
                    res_col = res_col + hist_hits(self.hists[col], op, val) - np.ones(n)
                ress[i, j * n : (j + 1) * n] = res_col
        return ress

    def extract_features(self, df: pd.DataFrame) -> TabularContainer:
        plan_to_operations, pred_triplets = self.extract_operations(
            df, get_pred_triplets
        )
        embeddings_list = self.hist_encoding(pred_triplets)
        op_emb = np.concatenate(
            [embeddings_list[plan_to_operations[idx]] for idx in df["id"].tolist()]
        )
        index_1 = np.concatenate(
            [
                np.array([plan] * len(operations))
                for plan, operations in plan_to_operations.items()
            ]
        )
        index_2 = np.concatenate(
            [np.arange(len(operations)) for _, operations in plan_to_operations.items()]
        ).astype(int)
        emb_df = pd.DataFrame(
            data=op_emb, columns=[f"emb_{i}" for i in range(op_emb.shape[1])]
        )
        emb_df["plan_id"] = index_1
        emb_df["operation_id"] = index_2
        emb_df = emb_df.set_index(["plan_id", "operation_id"])
        return TabularContainer(emb_df)


class PredicateBitmapExtractor(StaticExtractor[TabularContainer]):
    """Class to extract satisfiability bitmap of each predicate based on
    table samples.
    """

    def __init__(
        self,
        table_samples: Dict[str, pd.DataFrame],
        extract_operations: Callable[
            [pd.DataFrame, Callable], Tuple[Dict[int, List[int]], List[str]]
        ],
    ) -> None:
        super().__init__()
        if len(table_samples) == 0:
            raise ValueError("No table samples provided.")

        if np.std([df.shape[0] for df in table_samples.values()]) != 0:
            raise ValueError("All tables must have the same number of rows.")
        self.bitmap_size = np.mean(
            [df.shape[0] for df in table_samples.values()]
        ).astype(int)
        self.column2table = {
            col: table for table, df in table_samples.items() for col in df.columns
        }
        for table, df in table_samples.items():
            # also convert date columns to datetime64[ns] format
            for col in df.select_dtypes(include="object").columns:
                if isinstance(table_samples[table][col].iloc[0], date):
                    df[col] = pd.to_datetime(df[col])
            df["sid"] = np.arange(df.shape[0])
            table_samples[table] = df
        self.table_samples = table_samples
        self.extract_operations = extract_operations

    def bitmap_encoding(self, pred_triplets: List[str]) -> np.ndarray:
        n = self.bitmap_size
        ress = np.zeros((len(pred_triplets), n))
        for i, operations in enumerate(pred_triplets):
            if operations == "":
                continue
            triplets = [operation.split(",") for operation in operations.split(";")]
            if any(col not in self.column2table for col, _, _ in triplets):
                continue
            potential_tables = [self.column2table[col] for col, _, _ in triplets]
            if len(set(potential_tables)) != 1:
                raise ValueError("We have not implemented this case yet.")
            table = potential_tables[0]
            sql = " & ".join(
                [
                    f"""{col} {op if op != '=' else '=='} {("'" + val_str + "'")
                    if date_format_match(val_str) else val_str}"""
                    for col, op, val_str in triplets
                ]
            )
            bits = self.table_samples[table].query(sql).sid.values
            ress[i, bits] = 1
        return ress

    def extract_features(self, df: pd.DataFrame) -> TabularContainer:
        plan_to_operations, pred_triplets = self.extract_operations(
            df, get_pred_triplets
        )
        embeddings_list = self.bitmap_encoding(pred_triplets)
        op_emb = np.concatenate(
            [embeddings_list[plan_to_operations[idx]] for idx in df["id"].tolist()]
        )
        index_1 = np.concatenate(
            [
                np.array([plan] * len(operations))
                for plan, operations in plan_to_operations.items()
            ]
        )
        index_2 = np.concatenate(
            [np.arange(len(operations)) for _, operations in plan_to_operations.items()]
        ).astype(int)
        emb_df = pd.DataFrame(
            data=op_emb, columns=[f"emb_{i}" for i in range(op_emb.shape[1])]
        )
        emb_df["plan_id"] = index_1
        emb_df["operation_id"] = index_2
        emb_df = emb_df.set_index(["plan_id", "operation_id"])
        return TabularContainer(emb_df)
