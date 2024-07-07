"""Functions related to data, including creation, loading, and storage."""
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from sklearn.model_selection import train_test_split
from udao_spark.utils.collaborators import TypeAdvisor
from udao_trace.utils.handler import JsonHandler, ParquetHandler, PickleHandler

from udao.data.utils.query_plan import QueryPlanOperationFeatures, QueryPlanStructure

import numpy as np
import pandas as pd


def load_data(folder: Path, filename: str) -> pd.DataFrame:
    """Load query dataset as a Pandas DataFrame

    This function loads a dataset such as TPC-DS and returns it as a pandas DataFrame.
    Args:
        folder (Path): Location of the directory containing the dataset.
        filename (str): Name of the file containing the dataset.

    Returns:
        pd.DataFrame: Query dataset.
    """
    query_dataframe = ParquetHandler.load(folder, filename)
    return query_dataframe


def train_test_val_split_on_template_leave_out_fold(
    query_dataframe: pd.DataFrame,
    template_column: str,
    fold: int,
    n_folds: int,
    random_state: Optional[int] = None,
) -> Dict[pd.DataFrame, pd.DataFrame]:
    """Split dataset in training, validation and test sets.

    The split ensures that there are templates in the test set that have not
    been seen in the training and validation sets.

    Args:
        query_dataframe (pd.DataFrame): DataFrame containing the queries, their graph and their cost (latency, IO)
        template_column (str): Name of the column that contains the query template information.
        fold (int): Used to determine which templates are hidden from the training and validation sets.
        n_folds (int): Total number of folds.
        random_state (Optional[int], optional): Used to fix the seed of random generators for reproducibility purposes. Defaults to None.

    Returns:
        Dict[pd.DataFrame, pd.DataFrame]: `dict` containing three DataFrames, one for each split (train, val, test).
    """
    if fold not in range(1, n_folds):
        raise ValueError(f"fold must be in [1, 2, ... {n_folds}], got {fold}")

    # set seed for the function.
    np.random.seed(random_state)

    number_of_template_ids_per_fold : int = len(query_dataframe[template_column].unique()) // n_folds
    unique_template_ids = query_dataframe[template_column].unique()
    np.random.shuffle(unique_template_ids)

    if 1 <= fold < n_folds:
        test_template_ids = unique_template_ids[
            (fold - 1) * number_of_template_ids_per_fold : fold * number_of_template_ids_per_fold
            ]
    else:
        # this batch may not have the same size as the other folds.
        # This happens if the number of templates is not divisible by the number of folds.
        test_template_ids = unique_template_ids[
            (fold - 1) * number_of_template_ids_per_fold :
        ]

    templates_in_test_ids = np.isin(unique_template_ids, test_template_ids)
    # train and validation comprise templates not in the test set.
    train_validation_template_ids = unique_template_ids[~templates_in_test_ids]

    train_template_ids, val_template_ids = train_test_split(
        train_validation_template_ids,
        test_size=number_of_template_ids_per_fold,
        random_state=random_state
    )

    train_queries = query_dataframe[template_column].isin(train_template_ids)
    train_df = query_dataframe[train_queries]

    val_queries = query_dataframe[template_column].isin(val_template_ids)
    val_df = query_dataframe[val_queries]

    test_queries = query_dataframe[template_column].isin(test_template_ids)
    test_df = query_dataframe[test_queries]

    return {
        "train": train_df,
        "val": val_df,
        "test": test_df
    }
if __name__ == "__main__":
    # TODO(glachaud): this is some temporary place to store information while I work with the code.
    ta = TypeAdvisor(q_type="q_compile")
    data_folder = Path("cache_and_ckp/tpcds_perso")
    data_file = "df_q_compile.parquet"
    query_df = load_data(data_folder, data_file)