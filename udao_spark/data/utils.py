from itertools import chain
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch as th
from udao.data import BaseIterator
from udao.data.handler.data_handler import DataHandler
from udao.data.utils.utils import DatasetType, train_test_val_split_on_column

from udao_spark.utils.constants import (
    ALPHA_LQP_RAW,
    ALPHA_QS_RAW,
    BETA,
    BETA_RAW,
    EPS,
    GAMMA,
    THETA_C,
    THETA_P,
    THETA_RAW,
    THETA_S,
)
from udao_trace.configuration import SparkConf
from udao_trace.utils import BenchmarkType, JsonHandler, ParquetHandler, PickleHandler
from udao_trace.workload import Benchmark

from ..utils.collaborators import PathWatcher, TypeAdvisor
from ..utils.logging import logger
from ..utils.params import UdaoParams
from .handlers.data_processor import create_udao_data_processor


# Data Processing
def _im_process(df: pd.DataFrame) -> pd.DataFrame:
    df["IM-sizeInMB"] = df["IM-inputSizeInBytes"] / 1024 / 1024
    df["IM-sizeInMB-log"] = np.log(df["IM-sizeInMB"].to_numpy().clip(min=EPS))
    df["IM-rowCount"] = df["IM-inputRowCount"]
    df["IM-rowCount-log"] = np.log(df["IM-rowCount"].to_numpy().clip(min=EPS))
    for c in ALPHA_LQP_RAW:
        del df[c]
    return df


def extract_compile_time_im(graph_json_str: str) -> Tuple[float, float]:
    graph = JsonHandler.load_json_from_str(graph_json_str)
    operators, links = graph["operators"], graph["links"]
    outgoing_ids_set = set(link["toId"] for link in links)
    input_ids_set = set(range(len(operators))) - outgoing_ids_set
    im_size = sum(
        [
            operators[str(i)]["stats"]["compileTime"]["sizeInBytes"] / 1024.0 / 1024.0
            for i in input_ids_set
        ]
    )
    im_rows_count = sum(
        [
            operators[str(i)]["stats"]["compileTime"]["rowCount"] * 1.0
            for i in input_ids_set
        ]
    )
    return im_size, im_rows_count


def _im_process_compile(df: pd.DataFrame) -> pd.DataFrame:
    """a post-computation for compile-time input meta of each query stage"""
    df[["IM-sizeInMB-compile", "IM-rowCount-compile"]] = np.array(
        np.vectorize(extract_compile_time_im)(df["qs_lqp"])
    ).T
    df["IM-sizeInMB-compile-log"] = np.log(
        df["IM-sizeInMB-compile"].to_numpy().clip(min=EPS)
    )
    df["IM-rowCount-compile-log"] = np.log(
        df["IM-rowCount-compile"].to_numpy().clip(min=EPS)
    )
    return df


def extract_partition_distribution(pd_raw: str) -> Tuple[float, float, float]:
    pd = np.array(
        list(
            chain.from_iterable(
                JsonHandler.load_json_from_str(pd_raw.replace("'", '"')).values()
            )
        )
    )
    if pd.size == 0:
        return 0.0, 0.0, 0.0
    mu, std, max_val, min_val = np.mean(pd), np.std(pd), np.max(pd), np.min(pd)
    ratio1 = std / mu
    ratio2 = (max_val - mu) / mu
    ratio3 = (max_val - min_val) / mu
    return ratio1, ratio2, ratio3


def prepare_data(
    df: pd.DataFrame, sc: SparkConf, benchmark: str, q_type: str
) -> pd.DataFrame:
    bm = Benchmark(benchmark_type=BenchmarkType[benchmark.upper()])
    df.rename(columns={p: kid for p, kid in zip(THETA_RAW, sc.knob_ids)}, inplace=True)
    df["tid"] = df["template"].apply(lambda x: bm.get_template_id(str(x)))
    variable_names = sc.knob_ids
    theta = THETA_C + THETA_P + THETA_S
    if variable_names != theta:
        raise ValueError(f"variable_names != theta: {variable_names} != {theta}")
    df[variable_names] = sc.deconstruct_configuration(
        df[variable_names].astype(str).values
    )

    # extract alpha
    if q_type == "q":
        df[ALPHA_LQP_RAW] = df[ALPHA_LQP_RAW].astype(float)
        df = _im_process(df)
    elif q_type == "qs":
        df[ALPHA_QS_RAW] = df[ALPHA_QS_RAW].astype(float)
        df = _im_process(df)
        df = _im_process_compile(df)
        df["IM-init-part-num"] = df["InitialPartitionNum"].astype(float)
        df["IM-init-part-num-log"] = np.log(
            df["IM-init-part-num"].to_numpy().clip(min=EPS)
        )
        df.rename(columns={"total_task_duration_s": "ana_latency_s"}, inplace=True)
        df["ana_latency_s"] = df["ana_latency_s"] / df["k1"] / df["k3"]
        del df["InitialPartitionNum"]
    else:
        raise ValueError

    # extract beta
    df[BETA] = [
        extract_partition_distribution(pd_raw)
        for pd_raw in df[BETA_RAW].values.squeeze()
    ]
    for c in BETA_RAW:
        del df[c]

    # extract gamma:
    df[GAMMA] = df[GAMMA].astype(float)

    return df


def define_index_with_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    if "id" in df.columns:
        raise Exception("id column already exists!")
    df["id"] = df[columns].astype(str).apply("-".join, axis=1).to_list()
    df.set_index("id", inplace=True)
    return df


def save_and_log_index(index_splits: Dict, pw: PathWatcher, name: str) -> None:
    try:
        PickleHandler.save(index_splits, pw.cc_prefix, name)
    except FileExistsError as e:
        if not pw.debug:
            raise e
        logger.warning(f"skip saving {name}")
    lengths = [str(len(index_splits[split])) for split in ["train", "val", "test"]]
    logger.info(f"got index in {name}, tr/val/te={'/'.join(lengths)}")


def save_and_log_df(
    df: pd.DataFrame,
    index_columns: List[str],
    pw: PathWatcher,
    name: str,
) -> None:
    df = define_index_with_columns(df, columns=index_columns)
    try:
        ParquetHandler.save(df, pw.cc_prefix, f"{name}.parquet")
    except FileExistsError as e:
        if not pw.debug:
            raise e
        logger.warning(f"skip saving {name}.parquet")
    logger.info(f"prepared {name} shape: {df.shape}")


def checkpoint_model_structure(pw: PathWatcher, model_params: UdaoParams) -> str:
    model_struct_hash = model_params.hash()
    ckp_header = f"{pw.cc_extract_prefix}/{model_struct_hash}"
    json_name = "model_struct_params.json"
    if not Path(f"{ckp_header}/{json_name}").exists():
        JsonHandler.dump_to_file(
            model_params.to_dict(),
            f"{ckp_header}/{json_name}",
            indent=2,
        )
        logger.info(f"saved model structure params to {ckp_header}")
    else:
        logger.info(f"found {ckp_header}/{json_name}")
    return ckp_header


def magic_setup(pw: PathWatcher, seed: int) -> None:
    """magic set to make sure
    1. data has been properly processed and effectively saved.
    2. data split to make sure q_compile/q/qs share the same appid for tr/val/te.
    """
    df_q_raw = pd.read_csv(pw.get_data_header("q"), low_memory=pw.debug)
    df_qs_raw = pd.read_csv(pw.get_data_header("qs"), low_memory=pw.debug)
    logger.info(f"raw df_q shape: {df_q_raw.shape}")
    logger.info(f"raw df_qs shape: {df_qs_raw.shape}")

    benchmark = pw.benchmark
    debug = pw.debug

    # Prepare data
    sc = SparkConf(str(pw.base_dir / "assets/spark_configuration_aqe_on.json"))
    df_q = prepare_data(df_q_raw, benchmark=benchmark, sc=sc, q_type="q")
    df_qs = prepare_data(df_qs_raw, benchmark=benchmark, sc=sc, q_type="qs")
    df_q_compile = df_q[df_q["lqp_id"] == 0].copy()  # for compile-time df
    df_rare = df_q_compile.groupby("tid").filter(lambda x: len(x) < 5)
    if df_rare.shape[0] > 0:
        logger.warning(f"Drop rare templates: {df_rare['tid'].unique()}")
        df_q_compile = df_q_compile.groupby("tid").filter(lambda x: len(x) >= 5)
    else:
        logger.info("No rare templates")
    # Compute the index for df_q_compile, df_q and df_qs
    save_and_log_df(df_q_compile, ["appid"], pw, "df_q_compile")
    save_and_log_df(df_q, ["appid", "lqp_id"], pw, "df_q_all")
    save_and_log_df(df_qs, ["appid", "qs_id"], pw, "df_qs")

    # Split data for df_q_compile
    df_splits_q_compile = train_test_val_split_on_column(
        df=df_q_compile,
        groupby_col="tid",
        val_frac=0.2 if debug else 0.1,
        test_frac=0.2 if debug else 0.1,
        random_state=seed,
    )
    index_splits_q_compile = {
        split: df.index.to_list() for split, df in df_splits_q_compile.items()
    }
    index_splits_qs = {
        split: df_qs[df_qs.appid.isin(appid_list)].index.to_list()
        for split, appid_list in index_splits_q_compile.items()
    }
    index_splits_q = {
        split: df_q[df_q.appid.isin(appid_list)].index.to_list()
        for split, appid_list in index_splits_q_compile.items()
    }
    # Save the index_splits
    save_and_log_index(index_splits_q_compile, pw, "index_splits_q_compile.pkl")
    save_and_log_index(index_splits_q, pw, "index_splits_q_all.pkl")
    save_and_log_index(index_splits_qs, pw, "index_splits_qs.pkl")


# Data Split Index
def extract_index_splits(
    pw: PathWatcher, seed: int, q_type: str
) -> Tuple[pd.DataFrame, Dict[DatasetType, List[str]]]:
    if (
        not Path(f"{pw.cc_prefix}/index_splits_{q_type}.pkl").exists()
        or not Path(f"{pw.cc_prefix}/df_{q_type}.parquet").exists()
    ):
        logger.info(
            f"not found index_splits_{q_type}.pkl or df_{q_type}.parquet "
            f"under {pw.cc_prefix}, start magic setup..."
        )
        magic_setup(pw, seed)
    else:
        logger.info(f"found {pw.cc_prefix}/df_{q_type}.pkl, loading...")
        logger.info(f"found {pw.cc_prefix}/index_splits_{q_type}.pkl, loading...")

    if not Path(f"{pw.cc_prefix}/index_splits_{q_type}.pkl").exists():
        raise FileNotFoundError(f"{pw.cc_prefix}/index_splits_{q_type}.pkl not found")
    if not Path(f"{pw.cc_prefix}/df_{q_type}.parquet").exists():
        raise FileNotFoundError(f"{pw.cc_prefix}/df_{q_type}.parquet not found")

    df = ParquetHandler.load(pw.cc_prefix, f"df_{q_type}.parquet")
    index_splits = PickleHandler.load(pw.cc_prefix, f"index_splits_{q_type}.pkl")
    if not isinstance(index_splits, Dict):
        raise TypeError(f"index_splits is not a dict: {index_splits}")
    return df, index_splits


def extract_and_save_iterators(
    pw: PathWatcher,
    ta: TypeAdvisor,
    tensor_dtypes: th.dtype,
    cache_file: str = "split_iterators.pkl",
) -> Dict[DatasetType, BaseIterator]:
    params = pw.extract_params
    if Path(f"{pw.cc_extract_prefix}/{cache_file}").exists():
        raise FileExistsError(f"{pw.cc_extract_prefix}/{cache_file} already exists.")
    logger.info("start extracting split_iterators")
    df, index_splits = extract_index_splits(
        pw=pw, seed=params.seed, q_type=ta.get_q_type_for_cache()
    )

    cache_file_dp = "data_processor.pkl"
    if Path(f"{pw.cc_extract_prefix}/{cache_file_dp}").exists():
        raise FileExistsError(f"{pw.cc_extract_prefix}/{cache_file_dp} already exists.")
    logger.info("start creating data_processor")
    data_processor = create_udao_data_processor(
        ta=ta,
        sc=SparkConf(str(pw.base_dir / "assets/spark_configuration_aqe_on.json")),
        lpe_size=params.lpe_size,
        vec_size=params.vec_size,
        tensor_dtypes=tensor_dtypes,
    )
    data_handler = DataHandler(
        df.reset_index(),
        DataHandler.Params(
            index_column="id",
            stratify_on="tid",
            val_frac=0.2 if params.debug else 0.1,
            test_frac=0.2 if params.debug else 0.1,
            dryrun=False,
            data_processor=data_processor,
            random_state=params.seed,
        ),
    )
    data_handler.index_splits = index_splits
    logger.info("extracting split_iterators...")
    split_iterators = data_handler.get_iterators()

    PickleHandler.save(data_processor, pw.cc_extract_prefix, cache_file_dp)
    logger.info(f"saved {pw.cc_extract_prefix}/{cache_file_dp} after fitting")
    PickleHandler.save(split_iterators, pw.cc_extract_prefix, cache_file)
    logger.info(f"saved {pw.cc_extract_prefix}/{cache_file}")
    return split_iterators


def get_split_iterators(
    pw: PathWatcher,
    ta: TypeAdvisor,
    tensor_dtypes: th.dtype,
) -> Dict[DatasetType, BaseIterator]:
    cache_file = "split_iterators.pkl"
    if not Path(f"{pw.cc_extract_prefix}/{cache_file}").exists():
        return extract_and_save_iterators(
            pw=pw,
            ta=ta,
            tensor_dtypes=tensor_dtypes,
            cache_file=cache_file,
        )
    split_iterators = PickleHandler.load(pw.cc_extract_prefix, cache_file)
    if not isinstance(split_iterators, Dict):
        raise TypeError("split_iterators not found or not a desired type")
    return split_iterators


def get_lhs_confs(
    spark_conf: SparkConf, n_samples: int, seed: int, normalize: bool
) -> pd.DataFrame:
    lhs_conf_raw = spark_conf.get_lhs_configurations(n_samples, seed=seed)
    lhs_conf = pd.DataFrame(
        data=spark_conf.deconstruct_configuration(lhs_conf_raw.values),
        columns=spark_conf.knob_ids,
    )
    if normalize:
        theta_all_minmax = (
            np.array(spark_conf.knob_min),
            np.array(spark_conf.knob_max),
        )
        lhs_conf_norm = (lhs_conf - theta_all_minmax[0]) / (
            theta_all_minmax[1] - theta_all_minmax[0]
        )
        return lhs_conf_norm
    return lhs_conf
