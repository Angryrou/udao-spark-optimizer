from enum import Enum


class VarTypes(Enum):
    INT = "int"
    BOOL = "bool"
    CATEGORY = "category"
    FLOAT = "float"


class ScaleTypes(Enum):
    LOG = "log"
    LINEAR = "linear"


class BenchmarkType(Enum):
    TPCH = "tpch"
    TPCDS = "tpcds"
    TPCXBB = "tpcxbb"
    JOB_LIGHT = "job-light"
    JOB_SYNTHETIC = "job-synthetic"
    JOB_TRAIN = "job-train"
    JOB = "job"


class ClusterName(Enum):
    HEX1 = "hex1"
    HEX2 = "hex2"
    HEX3 = "hex3"
