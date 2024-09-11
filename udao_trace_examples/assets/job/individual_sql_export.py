import os

import pandas as pd

sql_path = "spark-sqls"
for prefix, name, path in [
    ("tr", "train", "./train.csv"),
    ("syn", "synthetic", "./synthetic.csv"),
    ("jl", "job-light", "./job-light.csv"),
]:
    os.makedirs(f"{sql_path}/{name}", exist_ok=True)
    df_train = pd.read_csv(path, sep="#", header=None, names=["c1", "c2", "c3", "c4"])
    for i, row in df_train.iterrows():
        tables = row["c1"]
        preds = []

        join_conditions = row["c2"]
        if isinstance(join_conditions, str):
            join_conditions = join_conditions.replace(",", " AND ")
            preds.append(join_conditions)

        col_predicates = row["c3"]
        if isinstance(col_predicates, str):
            col_predicates_strs = col_predicates.split(",")
            n_preds = len(col_predicates_strs) // 3
            col_predicates = " AND ".join(
                [
                    f"{col_predicates_strs[i * 3]}"
                    f"{col_predicates_strs[i * 3 + 1]}"
                    f"{col_predicates_strs[i * 3 + 2]}"
                    for i in range(n_preds)
                ]
            )
            preds.append(col_predicates)

        pred = " AND ".join(preds)
        sql = f"""SELECT COUNT(*) FROM {tables} WHERE {pred}"""

        with open(f"{sql_path}/{name}/{prefix}-{i}.sql", "w") as f:
            f.write(sql)
