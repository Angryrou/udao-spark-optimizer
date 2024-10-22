import random
from typing import Dict, List

# Define the database schema with min-max values
tables: Dict[str, Dict] = {
    "title": {
        "alias": "t",
        "columns": {
            "id": {"min": 1, "max": 2528312},
            "kind_id": {"min": 1, "max": 7},
            "production_year": {
                "min": 1950,
                "max": 2008,
            },  # shrink the range to concentrate on heavy hitter
        },
        "foreign_keys": [
            {"column": "id", "ref_table": "movie_companies", "ref_column": "movie_id"},
            {"column": "id", "ref_table": "cast_info", "ref_column": "movie_id"},
            {"column": "id", "ref_table": "movie_info", "ref_column": "movie_id"},
            {"column": "id", "ref_table": "movie_info_idx", "ref_column": "movie_id"},
            {"column": "id", "ref_table": "movie_keyword", "ref_column": "movie_id"},
        ],
    },
    "movie_info": {
        "alias": "mi",
        "columns": {
            "id": {"min": 1, "max": 14835720},
            "movie_id": {"min": 1, "max": 2526430},
            "info_type_id": {
                "min": 1,
                "max": 110,
                "heavy_hitter": [
                    16,
                    3,
                    7,
                    8,
                    4,
                    2,
                    1,
                    18,
                    15,
                    5,
                    6,
                    17,
                    98,
                    13,
                    107,
                    9,
                    105,
                    106,
                    94,
                    103,
                ],
            },
        },
        "foreign_keys": [
            {"column": "movie_id", "ref_table": "title", "ref_column": "id"}
        ],
    },
    "cast_info": {
        "alias": "ci",
        "columns": {
            "id": {"min": 1, "max": 36244344},
            "movie_id": {"min": 1, "max": 2525975},
            # "person_id": {"min": 1, "max": 4061926},
            "role_id": {"min": 1, "max": 11},
        },
        "foreign_keys": [
            {"column": "movie_id", "ref_table": "title", "ref_column": "id"}
        ],
    },
    "movie_companies": {
        "alias": "mc",
        "columns": {
            "id": {"min": 1, "max": 2609129},
            "movie_id": {"min": 2, "max": 2525745},
            "company_id": {
                "min": 1,
                "max": 234997,
                "heavy_hitter": [
                    6,
                    19,
                    160,
                    27,
                    166,
                    11137,
                    34,
                    11203,
                    11,
                    846,
                    2561,
                    1451,
                    11141,
                    225,
                    596,
                    424,
                    159,
                    7851,
                    302,
                    1284,
                    22956,
                ],
            },
            "company_type_id": {"min": 1, "max": 2},
        },
        "foreign_keys": [
            {"column": "movie_id", "ref_table": "title", "ref_column": "id"}
        ],
    },
    "movie_info_idx": {
        "alias": "mi_idx",
        "columns": {
            "id": {"min": 1, "max": 1380035},
            "movie_id": {"min": 2, "max": 2525793},
            "info_type_id": {"min": 99, "max": 113, "heavy_hitter": [99, 100, 101]},
        },
        "foreign_keys": [
            {"column": "movie_id", "ref_table": "title", "ref_column": "id"}
        ],
    },
    "movie_keyword": {
        "alias": "mk",
        "columns": {
            "id": {"min": 1, "max": 4523930},
            "movie_id": {"min": 2, "max": 2525971},
            "keyword_id": {
                "min": 1,
                "max": 134170,
                "heavy_hitter": [
                    335,
                    16264,
                    117,
                    2488,
                    359,
                    382,
                    137,
                    1,
                    1382,
                    121,
                    245,
                    347,
                    398,
                    870,
                    56,
                    236,
                    784,
                    875,
                    47,
                    3636,
                    7084,
                ],
            },
        },
        "foreign_keys": [
            {"column": "movie_id", "ref_table": "title", "ref_column": "id"}
        ],
    },
}

# Add reverse foreign keys for bidirectional traversal
for table_name, table_info in tables.items():
    table_info["reverse_foreign_keys"] = []

for table_name, table_info in tables.items():
    for fk in table_info["foreign_keys"]:
        ref_table = fk["ref_table"]
        tables[ref_table]["reverse_foreign_keys"].append(
            {
                "column": fk["ref_column"],
                "ref_table": table_name,
                "ref_column": fk["column"],
            }
        )


def generate_queries(query_cnt: int) -> List[str]:
    queries = []
    # fix the startign table as title
    start_table = "title"
    remaining_tables = [t for t in tables.keys() if t not in start_table]

    for _ in range(query_cnt):
        num_tables = random.choice([4, 5])  # 3-4 joins
        path = [start_table] + random.sample(remaining_tables, num_tables - 1)
        table_aliases = {table: tables[table]["alias"] for table in path}
        from_clause = ", ".join(
            [f"{table} {alias}" for table, alias in table_aliases.items()]
        )
        where_conditions = []
        # Build join conditions
        joins = []
        for i in range(len(path)):
            current_table = path[i]
            current_alias = table_aliases[current_table]
            for j in range(i + 1, len(path)):
                next_table = path[j]
                next_alias = table_aliases[next_table]
                # Check if there is a foreign key between current_table and next_table
                join_condition = None
                for fk in tables[current_table]["foreign_keys"]:
                    if fk["ref_table"] == next_table:
                        join_condition = (
                            f"{current_alias}.{fk['column']}"
                            f"={next_alias}.{fk['ref_column']}"
                        )
                        break
                for fk in tables[current_table].get("reverse_foreign_keys", []):
                    if fk["ref_table"] == next_table:
                        join_condition = (
                            f"{current_alias}.{fk['column']}"
                            f"={next_alias}.{fk['ref_column']}"
                        )
                        break
                if join_condition:
                    joins.append(join_condition)
        where_conditions.extend(joins)
        # Generate column predicates

        # 3 -> 2, 3, 4; 4 -> 3, 4
        num_predicates = random.randint(num_tables - 2, min(4, num_tables))
        for qi in range(num_predicates):
            # Randomly select a table
            pred_table = random.choice(path)
            pred_alias = table_aliases[pred_table]
            # Randomly select a column (excluding 'id')
            if pred_table == "title":
                columns = [
                    col for col in tables[pred_table]["columns"].keys() if col != "id"
                ]
            else:
                columns = [
                    col
                    for col in tables[pred_table]["columns"].keys()
                    if col not in ["id", "movie_id"]
                ]
            if not columns:
                continue
            column = random.choice(columns)
            # Use min-max values to generate appropriate value
            col_min = tables[pred_table]["columns"][column]["min"]
            col_max = tables[pred_table]["columns"][column]["max"]
            if pred_table == "title" and column == "production_year":
                # For 'production_year', use a range predicate
                operator = random.choice([">", "BETWEEN"])
                value = random.randint(col_min, col_max)
                if operator in [">"]:
                    predicate = f"{pred_alias}.{column}{operator}{value}"
                else:
                    if value > 2000:
                        dt = random.randint(4, 10)
                    else:
                        dt = random.choice([5, 10, 20, 50])
                    predicate = (
                        f"{pred_alias}.{column}>{value} "
                        f"AND {pred_alias}.{column}<{value + dt}"
                    )
            else:
                # For other columns, use equality predicates
                if "heavy_hitter" in tables[pred_table]["columns"][column]:
                    heavy_hitter = tables[pred_table]["columns"][column]["heavy_hitter"]
                    value = random.choice(heavy_hitter)
                else:
                    value = random.randint(col_min, col_max)
                predicate = f"{pred_alias}.{column}={value}"
            where_conditions.append(predicate)
        # Build the query
        where_clause = " AND ".join(where_conditions)
        query = f"SELECT COUNT(*) FROM {from_clause} WHERE {where_clause}"
        queries.append(query)
    return queries


# Example usage
query_cnt = 40000  # Replace with your desired number of queries
random.seed(0)
queries = generate_queries(query_cnt)

sql_path = "spark-sqls"
name = "job-ext"
with open(f"{sql_path}/{name}_{query_cnt}.txt", "w") as f:
    f.write("\n".join(queries))

# os.makedirs(f"{sql_path}/{name}", exist_ok=True)
# for qid, q in enumerate(queries):
#     with open(f"{sql_path}/{name}/{qid}.sql", "w") as f:
#         f.write(q)
