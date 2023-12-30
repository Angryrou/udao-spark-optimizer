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
