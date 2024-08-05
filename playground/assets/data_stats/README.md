Steps to get hist of each numerical column and samples for bitmap.

1. turn on a spark-shell
  ```bash
  cd ~/spark
  jpath=/opt/hex_users/$USER/spark-3.2.1-hadoop3.3.0/jdk1.8
  ~/spark/bin/spark-shell \
  --master yarn \
  --deploy-mode client \
  --conf spark.executorEnv.JAVA_HOME=$jpath \
  --conf spark.yarn.appMasterEnv.JAVA_HOME=$jpath \
  --conf spark.default.parallelism=140 \
  --conf spark.executor.instances=35 \
  --conf spark.executor.cores=4 \
  --conf spark.executor.memory=80g \
  --conf spark.driver.memory=80g \
  --conf spark.reducer.maxSizeInFlight=256m \
  --conf spark.rpc.askTimeout=12000 \
  --conf spark.shuffle.io.retryWait=60 \
  --conf spark.sql.autoBroadcastJoinThreshold=200m \
  --conf "spark.driver.extraJavaOptions=-Xms20g" \
  --jars /opt/hex_users/$USER/chenghao/spark-sql-perf/target/scala-2.12/spark-sql-perf_2.12-0.5.1-SNAPSHOT.jar
  ```

2. export histograms for each column in tables from spark-shell
```scala
val benchmark = "tpch"
sql(s"use ${benchmark}_100")
val tables = spark.catalog.listTables().collect()

import java.time.LocalDate
import java.time.format.DateTimeFormatter

// Function to convert day offsets to date strings
def convertDaysToDate(days: Int, epochStartDate: LocalDate): String = {
  val targetDate = epochStartDate.plusDays(days)
  targetDate.format(DateTimeFormatter.ofPattern("yyyy-MM-dd"))
}

var allHistograms = Seq[(String, String, String, String, Int)]()
val epochStartDate = LocalDate.parse("1970-01-01")
tables.foreach { table =>
  spark.table(table.name).columns.foreach { columnName =>
    val desc = spark.sql(s"DESCRIBE EXTENDED ${table.name} $columnName").collect()
    val dataType = desc.find(_.getString(0) == "data_type").map(_.getString(1)).getOrElse("N/A")
    val distinctCount = desc.find(_.getString(0) == "distinct_count").map(_.getString(1).toInt).getOrElse(-1)

    val bins = desc.filter(row => row.getString(0).startsWith("bin_"))
    if (bins.nonEmpty) {
      val boundaries = bins.flatMap { row =>
        val binDetails = row.getString(1).split(",").map(_.trim)
        val lowerBound = binDetails(0).split(":")(1).trim
        val upperBound = binDetails(1).split(":")(1).trim
        val formattedLower = if (dataType == "date") convertDaysToDate(lowerBound.toDouble.toInt, epochStartDate) else lowerBound
        val formattedUpper = if (dataType == "date") convertDaysToDate(upperBound.toDouble.toInt, epochStartDate) else upperBound
        Seq(formattedLower, formattedUpper)
      }.distinct

      val boundaryList = boundaries.mkString("[", ", ", "]")
      allHistograms :+= (table.name, columnName, dataType, boundaryList, distinctCount)
    } else {
      allHistograms :+= (table.name, columnName, dataType, "[]", distinctCount)
    }
  }
}

// Convert the sequence to a DataFrame
val histDF = spark.createDataFrame(allHistograms).toDF("table", "column", "dtype", "bins", "distinct_count")
histDF.show(false)

// Write the DataFrame to a CSV file
val path = "/user/hex1/raw_tpch_hist.csv"
// val path = "/user/hex2/raw_tpcds_hist.csv"
histDF.coalesce(1).write.option("header", "true").csv(path)
```

3. copy the csv file to local
```bash
# on hex1@node1
hadoop fs -getmerge /user/hex1/raw_tpch_hist.csv ~/chenghao/raw_tpch_hist.csv
# on hex2@node7
hadoop fs -getmerge /user/hex2/raw_tpcds_hist.csv ~/chenghao/raw_tpcds_hist.csv
```

4. trigger pyspark to collect table row samples for bitmap construction
```bash
cd ~/spark
jpath=/opt/hex_users/$USER/spark-3.2.1-hadoop3.3.0/jdk1.8
~/spark/bin/pyspark \
  --master yarn \
  --deploy-mode client \
  --conf spark.executorEnv.JAVA_HOME=$jpath \
  --conf spark.yarn.appMasterEnv.JAVA_HOME=$jpath \
  --conf spark.default.parallelism=140 \
  --conf spark.executor.instances=35 \
  --conf spark.executor.cores=4 \
  --conf spark.executor.memory=80g \
  --conf spark.driver.memory=80g \
  --conf spark.reducer.maxSizeInFlight=256m \
  --conf spark.rpc.askTimeout=12000 \
  --conf spark.shuffle.io.retryWait=60 \
  --conf spark.sql.autoBroadcastJoinThreshold=200m \
  --conf "spark.driver.extraJavaOptions=-Xms20g" \
  --jars /opt/hex_users/$USER/chenghao/spark-sql-perf/target/scala-2.12/spark-sql-perf_2.12-0.5.1-SNAPSHOT.jar
```

5. expose table samples within the pyspark session.
```python
# within pyspark
database_name = "tpch_100" # or "tpcds_100"
sql(f"USE {database_name}")
tables = sql(f"SHOW TABLES").toPandas()
table_names = tables['tableName'].tolist()

def get_table_row_count(table_name):
    # Collect all rows to the driver
    describe_rows = sql(f"DESCRIBE FORMATTED {table_name}").collect()

    # Filter to find the row with statistics information
    stats_row = list(filter(lambda row: 'Statistics' in row[0], describe_rows))

    # Extract the number of rows from the statistics row if it exists
    if stats_row:
        stats_info = stats_row[0][1]
        num_rows = int(stats_info.split(',')[1].split()[0])  # Parses the "30 rows" part
        return num_rows
    else:
        print(f"Table {table_name} does not have row count information.")
        return -1

result = {}
for table_name in table_names:
    row_count = get_table_row_count(table_name)
    df = sql(f"SELECT * FROM {table_name}")
    pd_df = df.sample(withReplacement=True, fraction=1200 / row_count).toPandas()
    if pd_df.shape[0] < 1000:
        raise ValueError(f"Table {table_name} has less than 1000 rows, only {result[table_name].shape[0]} rows are sampled.")
    result[table_name] = pd_df.sample(n=1000)

result = {
    table_name: result[table_name].reset_index(drop=True)
    for table_name in result
}

# Save the samples to a pickle file
import pickle
with open(f"/opt/hex_users/hex1/chenghao/{database_name}_samples.pkl", "wb") as f:
    pickle.dump(result, f)
# # for tpcds_100
# with open(f"/opt/hex_users/hex2/chenghao/{database_name}_samples.pkl", "wb") as f:
#     pickle.dump(result, f)
```
