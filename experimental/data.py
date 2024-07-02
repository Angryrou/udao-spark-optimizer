from pathlib import Path
from udao_trace.utils.handler import ParquetHandler

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


