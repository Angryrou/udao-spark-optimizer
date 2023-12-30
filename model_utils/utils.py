from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch as th
from sklearn.preprocessing import MinMaxScaler
from udao.data import (
    BaseIterator,
    DataProcessor,
    NormalizePreprocessor,
    PredicateEmbeddingExtractor,
    QueryPlanIterator,
    QueryStructureContainer,
    QueryStructureExtractor,
    TabularFeatureExtractor,
)
from udao.data.handler.data_handler import DataHandler
from udao.data.handler.data_processor import FeaturePipeline, create_data_processor
from udao.data.predicate_embedders import Word2VecEmbedder, Word2VecParams
from udao.data.predicate_embedders.utils import build_unique_operations
from udao.data.utils.query_plan import QueryPlanOperationFeatures, QueryPlanStructure
from udao.data.utils.utils import DatasetType, train_test_val_split_on_column
from udao.utils.logging import logger

from udao_trace.configuration import SparkConf
from udao_trace.utils import BenchmarkType, JsonHandler, ParquetHandler, PickleHandler
from udao_trace.workload import Benchmark

from .params import ExtractParams

tensor_dtypes = th.float32

THETA_RAW = [
    "theta_c-spark.executor.cores",
    "theta_c-spark.executor.memory",
    "theta_c-spark.executor.instances",
    "theta_c-spark.default.parallelism",
    "theta_c-spark.reducer.maxSizeInFlight",
    "theta_c-spark.shuffle.sort.bypassMergeThreshold",
    "theta_c-spark.shuffle.compress",
    "theta_c-spark.memory.fraction",
    "theta_p-spark.sql.adaptive.advisoryPartitionSizeInBytes",
    "theta_p-spark.sql.adaptive.nonEmptyPartitionRatioForBroadcastJoin",
    "theta_p-spark.sql.adaptive.maxShuffledHashJoinLocalMapThreshold",
    "theta_p-spark.sql.adaptive.autoBroadcastJoinThreshold",
    "theta_p-spark.sql.shuffle.partitions",
    "theta_p-spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes",
    "theta_p-spark.sql.adaptive.skewJoin.skewedPartitionFactor",
    "theta_p-spark.sql.files.maxPartitionBytes",
    "theta_p-spark.sql.files.openCostInBytes",
    "theta_s-spark.sql.adaptive.rebalancePartitionsSmallPartitionFactor",
    "theta_s-spark.sql.adaptive.coalescePartitions.minPartitionSize",
]
ALPHA_LQP_RAW = [
    "IM-inputSizeInBytes",
    "IM-inputRowCount",
]
ALPHA_QS_RAW = ["InitialPartitionNum"] + ALPHA_LQP_RAW
BETA_RAW = ["PD"]
THETA = [
    "k1",
    "k2",
    "k3",
    "k4",
    "k5",
    "k6",
    "k7",
    "k8",
    "s1",
    "s2",
    "s3",
    "s4",
    "s5",
    "s6",
    "s7",
    "s8",
    "s9",
    "s10",
    "s11",
]
ALPHA_LQP = [
    "IM-sizeInMB",
    "IM-rowCount",
    "IM-sizeInMB-log",
    "IM-rowCount-log",
]
ALPHA_QS = ["IM-init-part-num", "IM-init-part-num-log"] + ALPHA_LQP
BETA = ["PD-std-avg", "PD-skewness-ratio", "PD-range-avg-ratio"]
GAMMA = [
    "SS-RunningTasksNum",
    "SS-FinishedTasksNum",
    "SS-FinishedTasksTotalTimeInMs",
    "SS-FinishedTasksDistributionInMs-0tile",
    "SS-FinishedTasksDistributionInMs-25tile",
    "SS-FinishedTasksDistributionInMs-50tile",
    "SS-FinishedTasksDistributionInMs-75tile",
    "SS-FinishedTasksDistributionInMs-100tile",
]
TABULAR_LQP = THETA + ALPHA_LQP + BETA + GAMMA
TABULAR_QS = THETA + ALPHA_QS + BETA + GAMMA

EPS = 1e-3


class NoBenchmarkError(ValueError):
    """raise when no valid benchmark is found"""


class NoQTypeError(ValueError):
    """raise when no valid mode is found (only q and qs)"""


class OperatorMisMatchError(BaseException):
    """raise when the operator names from `operator` and `link` do not match"""


class PathWatcher:
    def __init__(
        self, base_dir: Path, benchmark: str, debug: bool, extract_params: ExtractParams
    ):
        self.base_dir = base_dir
        self.benchmark = benchmark
        self.debug = debug
        data_sign = self._get_data_sign()
        data_prefix = f"{str(base_dir)}/data/{benchmark}"
        cc_prefix = f"{str(base_dir)}/cache_and_ckp/{benchmark}_{data_sign}"
        cc_extract_prefix = f"{cc_prefix}/{extract_params.hash()}"
        self.data_sign = data_sign
        self.data_prefix = data_prefix
        self.cc_prefix = cc_prefix
        self.cc_extract_prefix = cc_extract_prefix
        self.extract_params = extract_params
        self._checkpoint_split()

    def _checkpoint_split(self) -> None:
        json_name = "extract_param.json"
        if not Path(f"{self.cc_extract_prefix}/{json_name}").exists():
            JsonHandler.dump_to_file(
                self.extract_params.__dict__,
                f"{self.cc_extract_prefix}/{json_name}",
                indent=2,
            )
            logger.info(f"saved split params to {self.cc_extract_prefix}/{json_name}")
        else:
            logger.info(f"found {self.cc_extract_prefix}/{json_name}")

    def _get_data_sign(self) -> str:
        # Read data
        if self.benchmark == "tpch":
            return f"22x{10 if self.debug else 2273}"
        if self.benchmark == "tpcds":
            return f"102x{10 if self.debug else 490}"
        raise NoBenchmarkError

    def get_data_header(self, q_type: str) -> str:
        return f"{self.data_prefix}/{q_type}_{self.data_sign}.csv"


# Data Processing
def _im_process(df: pd.DataFrame) -> pd.DataFrame:
    df["IM-sizeInMB"] = df["IM-inputSizeInBytes"] / 1024 / 1024
    df["IM-sizeInMB-log"] = np.log(df["IM-sizeInMB"].to_numpy().clip(min=EPS))
    df["IM-rowCount"] = df["IM-inputRowCount"]
    df["IM-rowCount-log"] = np.log(df["IM-rowCount"].to_numpy().clip(min=EPS))
    for c in ALPHA_LQP_RAW:
        del df[c]
    return df


def prepare_data(
    df: pd.DataFrame, sc: SparkConf, benchmark: str, q_type: str
) -> pd.DataFrame:
    bm = Benchmark(benchmark_type=BenchmarkType[benchmark.upper()])
    df.rename(columns={p: kid for p, kid in zip(THETA_RAW, sc.knob_ids)}, inplace=True)
    df["tid"] = df["template"].apply(lambda x: bm.get_template_id(str(x)))
    variable_names = sc.knob_ids
    if variable_names != THETA:
        raise ValueError(f"variable_names != THETA: {variable_names} != {THETA}")
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
        df["IM-init-part-num"] = df["InitialPartitionNum"].astype(float)
        df["IM-init-part-num-log"] = np.log(
            df["IM-init-part-num"].to_numpy().clip(min=EPS)
        )
        del df["InitialPartitionNum"]
    else:
        raise NoQTypeError

    # extract beta
    df[BETA] = [
        sc.extract_partition_distribution(pd_raw)
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


def magic_setup(pw: PathWatcher, seed: int) -> None:
    """magic set to make sure
    1. data has been properly processed and effectively saved.
    2. data split to make sure q_compile/q/qs share the same appid for tr/val/te.
    """
    df_q_raw = pd.read_csv(pw.get_data_header("q"))
    df_qs_raw = pd.read_csv(pw.get_data_header("qs"))
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
    save_and_log_df(df_q, ["appid", "lqp_id"], pw, "df_q")
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
    save_and_log_index(index_splits_q, pw, "index_splits_q.pkl")
    save_and_log_index(index_splits_qs, pw, "index_splits_qs.pkl")


def define_data_processor(
    lpe_size: int,
    vec_size: int,
    tabular_columns: List[str],
    objectives: List[str],
) -> DataProcessor:
    data_processor_getter = create_data_processor(QueryPlanIterator, "op_enc")
    return data_processor_getter(
        tensor_dtypes=tensor_dtypes,
        tabular_features=FeaturePipeline(
            extractor=TabularFeatureExtractor(columns=tabular_columns),
            preprocessors=[NormalizePreprocessor(MinMaxScaler())],
        ),
        objectives=FeaturePipeline(
            extractor=TabularFeatureExtractor(columns=objectives),
        ),
        query_structure=FeaturePipeline(
            extractor=LQPExtractor(positional_encoding_size=lpe_size),
            preprocessors=[NormalizePreprocessor(MinMaxScaler(), "graph_features")],
        ),
        op_enc=FeaturePipeline(
            extractor=PredicateEmbeddingExtractor(
                Word2VecEmbedder(Word2VecParams(vec_size=vec_size)),
                extract_operations=extract_operations_from_serialized_json,
            ),
        ),
    )


# Data Split Index
def extract_index_splits(
    pw: PathWatcher, seed: int, q_type: str
) -> Tuple[pd.DataFrame, Dict[DatasetType, List[str]]]:
    if (
        not Path(f"{pw.cc_prefix}/index_splits_{q_type}.pkl").exists()
        or not Path(f"{pw.cc_prefix}/df_{q_type}.pkl").exists()
    ):
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
    params: ExtractParams,
    tabular_columns: List[str],
    objectives: List[str],
    cache_file: str = "iterators.pkl",
) -> Dict[DatasetType, BaseIterator]:
    if Path(f"{pw.cc_extract_prefix}/{cache_file}").exists():
        raise FileExistsError(f"{pw.cc_extract_prefix}/{cache_file} already exists.")
    logger.info("start extracting iterators")
    cache_file_dh = "data_handler.pkl"
    if Path(f"{pw.cc_extract_prefix}/{cache_file_dh}").exists():
        logger.info(f"found {pw.cc_extract_prefix}/{cache_file_dh}, loading...")
        data_handler = PickleHandler.load(pw.cc_extract_prefix, cache_file_dh)
        if not isinstance(data_handler, DataHandler):
            raise TypeError(f"data_handler is not a DataHandler: {data_handler}")
    else:
        logger.info(f"not found {pw.cc_extract_prefix}/{cache_file_dh}, extracting...")
        df, index_splits = extract_index_splits(
            pw=pw, seed=params.seed, q_type=params.q_type
        )
        data_processor = define_data_processor(
            lpe_size=params.lpe_size,
            vec_size=params.vec_size,
            tabular_columns=tabular_columns,
            objectives=objectives,
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
        if not isinstance(data_handler, DataHandler):
            raise TypeError(f"data_handler is not a DataHandler: {data_handler}")
        data_handler.index_splits = index_splits
        PickleHandler.save(data_handler, pw.cc_extract_prefix, cache_file_dh)
        logger.info(f"saved {pw.cc_extract_prefix}/{cache_file_dh}")
    logger.info("extracting split_iterators...")
    split_iterators = data_handler.get_iterators()
    PickleHandler.save(
        {"desc": params.__dict__, "split_iterators": split_iterators},
        pw.cc_extract_prefix,
        cache_file,
    )
    logger.info(f"saved {pw.cc_extract_prefix}/{cache_file}")
    return split_iterators


def get_split_iterators(
    pw: PathWatcher,
    params: ExtractParams,
    tabular_columns: List[str],
    objectives: List[str],
) -> Dict[DatasetType, BaseIterator]:
    cache_file = "iterators.pkl"
    if not Path(f"{pw.cc_extract_prefix}/{cache_file}").exists():
        split_iterators = extract_and_save_iterators(
            pw=pw,
            params=params,
            tabular_columns=tabular_columns,
            objectives=objectives,
            cache_file=cache_file,
        )
        return split_iterators
    split_meta = PickleHandler.load(pw.cc_extract_prefix, cache_file)
    if (
        not isinstance(split_meta, Dict)
        or "split_iterators" not in split_meta
        or "desc" not in split_meta
        or not isinstance(split_meta["split_iterators"], Dict)
    ):
        raise TypeError("split_iterators not found or not a desired type")
    logger.info(split_meta["desc"])
    return split_meta["split_iterators"]


def extract_operations_from_serialized_json(
    plan_df: pd.DataFrame, operation_processing: Callable[[str], str] = lambda x: x
) -> Tuple[Dict[int, List[int]], List[str]]:
    df = plan_df[["id", "lqp"]].copy()
    df["lqp"] = df["lqp"].apply(
        lambda lqp_str: [
            operation_processing(op["predicate"])
            for op_id, op in JsonHandler.load_json_from_str(lqp_str)[
                "operators"
            ].items()
        ]  # type: ignore
    )
    df = df.explode("lqp", ignore_index=True)
    df.rename(columns={"lqp": "operation"}, inplace=True)
    return build_unique_operations(df)


def extract_query_plan_features_from_serialized_json(
    lqp_str: str,
) -> Tuple[QueryPlanStructure, QueryPlanOperationFeatures]:
    lqp = JsonHandler.load_json_from_str(lqp_str)
    operators, links = lqp["operators"], lqp["links"]
    num_operators = len(operators)
    id2name = {
        int(op_id): op["className"].split(".")[-1] for op_id, op in operators.items()
    }
    incoming_ids: List[int] = []
    outgoing_ids: List[int] = []
    for link in links:
        from_id, to_id = link["fromId"], link["toId"]
        if link["fromName"] != id2name[from_id] or link["toName"] != id2name[to_id]:
            raise OperatorMisMatchError
        incoming_ids.append(from_id)
        outgoing_ids.append(to_id)
    node_names = [id2name[i] for i in range(num_operators)]
    sizes = [
        np.log(
            np.clip(
                operators[str(i)]["sizeInBytes"] / 1024.0 / 1024.0,
                a_min=EPS,
                a_max=None,
            )
        )
        for i in range(num_operators)
    ]
    rows_counts = [
        np.log(np.clip(operators[str(i)]["rowCount"] * 1.0, a_min=EPS, a_max=None))
        for i in range(num_operators)
    ]
    op_features = QueryPlanOperationFeatures(rows_count=rows_counts, size=sizes)
    structure = QueryPlanStructure(
        node_names=node_names, incoming_ids=incoming_ids, outgoing_ids=outgoing_ids
    )
    return structure, op_features


class LQPExtractor(QueryStructureExtractor):
    def __init__(self, positional_encoding_size: Optional[int] = None):
        super(LQPExtractor, self).__init__(positional_encoding_size)

    def _extract_structure_and_features(
        self, idx: str, lqp: str, split: DatasetType
    ) -> Dict:
        structure, op_features = extract_query_plan_features_from_serialized_json(lqp)
        operation_gids = self._extract_operation_types(structure, split)
        self.id_template_dict[idx] = self._extract_structure_template(structure, split)
        return {
            "operation_id": op_features.operation_ids,
            "operation_gid": operation_gids,
            **op_features.features_dict,
        }

    def extract_features(
        self, df: pd.DataFrame, split: DatasetType
    ) -> QueryStructureContainer:
        df_op_features: pd.DataFrame = df.apply(
            lambda row: self._extract_structure_and_features(row.id, row.lqp, split),
            axis=1,
        ).apply(pd.Series)
        df_op_features["plan_id"] = df["id"]
        (
            df_op_features_exploded,
            df_operation_types,
        ) = self._extract_op_features_exploded(df_op_features)
        return QueryStructureContainer(
            graph_features=df_op_features_exploded,
            template_plans=self.template_plans,
            key_to_template=self.id_template_dict,
            graph_meta_features=None,
            operation_types=df_operation_types,
        )
