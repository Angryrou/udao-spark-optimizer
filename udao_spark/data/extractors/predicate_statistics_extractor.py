import re
from datetime import date, datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from udao.data import StaticExtractor, TabularContainer
from udao.data.predicate_embedders.utils import build_unique_operations

from udao_trace.utils import JsonHandler


def get_table_metadata(pred_str: str) -> Dict[str, str]:
    columns_pattern = r"\[([^\]]+)\]"
    columns_match = re.search(columns_pattern, pred_str)
    if columns_match:
        columns_str = columns_match.group(1)
        columns_list = [col.strip() for col in columns_str.split(",")]
    else:
        raise Exception(f"Columns not found in {pred_str}")

    table_pattern = r"`[^`]+`\.`([^`]+)`\.`([^`]+)`"
    table_match = re.search(table_pattern, pred_str)
    if table_match:
        table_name = table_match.group(1) + "." + table_match.group(2)
    else:
        raise Exception(f"Table name not found in {pred_str}")
    return {col: table_name for col in columns_list}


def get_pred_triplets_with_meta(pred: str, col2rel: Dict[str, str]) -> str:
    """
    Extract triplets for hist/bitmaps encoding.
    Rules:
        0. extract any <col, op, constant> triplets.
        1. join predicate will not be matched.
        2. predicates joined by AND is supported.
        3. a predicate with OR is not supported. (*)
        4. a predicated with CASE WHEN is not supported. (*)

    (*) easy to extract but adapting to the hist/bitmaps encoder can be complex.
        We consider a postponed improvement in the future.
    """
    if "CASE WHEN" in pred or "OR" in pred:
        return ""
    pattern = (
        r"(\w+#\d+)\s*([<>=!]{1,2})\s*(\d{4}-\d{2}-\d{2}|[-]?\d+\.\d+"
        r"|[A-Za-z0-9\s_.?!+-]+(?!\S*#\d+\b))"
    )
    matches = re.finditer(pattern, pred)  # type: ignore
    triplets = [match.groups() for match in matches]
    triplets = [triplet for triplet in triplets if triplet[0] in col2rel]
    rel_set = set([col2rel[triplet[0]] for triplet in triplets])
    if len(rel_set) == 0:
        return ""
    if len(rel_set) > 1:
        raise Exception(f"Multiple tables found in {pred}")
    else:
        rel = rel_set.pop()
    triplets_str = (
        rel
        + ":"
        + ";".join(
            [
                ",".join([triplet[0].split("#")[0], triplet[1], triplet[2]])
                for triplet in triplets
            ]
        )
    )
    return triplets_str


def get_pred_triplets_with_table_name(lqp_str: str) -> List[str]:
    operators = JsonHandler.load_json_from_str(lqp_str)["operators"]
    col2rel = {}
    for v in operators.values():
        if "LogicalRelation" in v["className"]:
            pred_str = v["predicate"]
            col2rel.update(get_table_metadata(pred_str))
    id2triplets = {
        int(op_id): get_pred_triplets_with_meta(op["predicate"], col2rel)
        for op_id, op in operators.items()
    }
    return [id2triplets[i] for i in range(len(id2triplets))]


def extract_operations_with_table_names(
    df: pd.DataFrame,
) -> Tuple[Dict[int, List[int]], List[str]]:
    graph_column = "lqp"
    df = df[["id", graph_column]].copy()
    df[graph_column] = df[graph_column].apply(
        lambda x: get_pred_triplets_with_table_name(x)
    )
    df = df.explode(graph_column, ignore_index=True)
    df.rename(columns={graph_column: "operation"}, inplace=True)
    return build_unique_operations(df)


def date_format_match(val_str: str) -> bool:
    if re.match(r"^\d{4}-\d{2}-\d{2}$", val_str):
        return True
    try:
        float(val_str)
        return False
    except Exception:
        return True


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
        hists: Dict[Tuple[str, str], np.ndarray],
        max_num_of_column_predicates: int = 3,
    ):
        super().__init__()
        if len(hists) == 0:
            raise ValueError("No histograms provided.")
        self.hists = hists
        self.hists_bins = len(list(hists.values())[0]) - 1  # 50
        self.max_triplets = max_num_of_column_predicates  # 3

    def hist_encoding(self, pred_triplets: List[str]) -> np.ndarray:
        n = self.hists_bins
        ress = np.zeros((len(pred_triplets), n * self.max_triplets))
        for i, operations in enumerate(pred_triplets):
            if operations == "":
                continue
            table_name, triplet_str = operations.split(":")
            triplets = [operation.split(",") for operation in triplet_str.split(";")]
            if any((table_name, triplet[0]) not in self.hists for triplet in triplets):
                continue

            triplets_dict = {}
            for operation in triplet_str.split(";"):
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
                    res_col = (
                        res_col
                        + hist_hits(self.hists[(table_name, col)], op, val)
                        - np.ones(n)
                    )
                ress[i, j * n : (j + 1) * n] = res_col
        return ress

    def extract_features(self, df: pd.DataFrame) -> TabularContainer:
        plan_to_operations, pred_triplets = extract_operations_with_table_names(df)
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
    ) -> None:
        super().__init__()
        if len(table_samples) == 0:
            raise ValueError("No table samples provided.")

        if np.std([df.shape[0] for df in table_samples.values()]) != 0:
            raise ValueError("All tables must have the same number of rows.")
        self.bitmap_size = np.mean(
            [df.shape[0] for df in table_samples.values()]
        ).astype(int)
        for table, df in table_samples.items():
            # also convert date columns to datetime64[ns] format
            for col in df.select_dtypes(include="object").columns:
                if isinstance(table_samples[table][col].iloc[0], date):
                    df[col] = pd.to_datetime(df[col])
            df["sid"] = np.arange(df.shape[0])
            table_samples[table] = df
        self.table_samples = table_samples

    def bitmap_encoding(self, pred_triplets: List[str]) -> np.ndarray:
        n = self.bitmap_size
        ress = np.zeros((len(pred_triplets), n))
        for i, operations in enumerate(pred_triplets):
            if operations == "":
                continue
            table_name, triplets_str = operations.split(":")
            triplets = [operation.split(",") for operation in triplets_str.split(";")]
            sql = " & ".join(
                [
                    f"""{col} {op if op != '=' else '=='} {("'" + val_str + "'")
                    if date_format_match(val_str) else val_str}"""
                    for col, op, val_str in triplets
                ]
            )
            bits = self.table_samples[table_name].query(sql).sid.values
            ress[i, bits] = 1
        return ress

    def extract_features(self, df: pd.DataFrame) -> TabularContainer:
        # plan_to_operations, pred_triplets = self.extract_operations(
        #     df, get_pred_triplets
        # )
        plan_to_operations, pred_triplets = extract_operations_with_table_names(df)
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
