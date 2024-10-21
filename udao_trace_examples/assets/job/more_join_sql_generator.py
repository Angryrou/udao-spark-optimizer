import os.path
import random
from typing import Dict, List

# Define the database schema with min-max values
tables: Dict[str, Dict] = {
    "title": {
        "alias": "t",
        "columns": {
            "id": {"min": 1, "max": 2528312},
            "kind_id": {"min": 1, "max": 7},
            "production_year": {"min": 1880, "max": 2019},
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
            "info_type_id": {"min": 1, "max": 110},
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
            "person_id": {"min": 1, "max": 4061926},
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
            "company_id": {"min": 1, "max": 234997},
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
            "info_type_id": {"min": 99, "max": 113},
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
            "keyword_id": {"min": 1, "max": 134170},
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
                    if col != "movie_id"
                ]
            if not columns:
                continue
            column = random.choice(columns)
            # Use min-max values to generate appropriate value
            col_min = tables[pred_table]["columns"][column]["min"]
            col_max = tables[pred_table]["columns"][column]["max"]
            if pred_table == "title" and column == "production_year":
                # For 'production_year', use a range predicate
                operator = random.choice([">", "<", "BETWEEN"])
                if operator in [">", "<"]:
                    value = random.randint(col_min, col_max)
                    predicate = f"{pred_alias}.{column}{operator}{value}"
                else:
                    all_choices = list(range(col_min, col_max + 1))
                    v1, v2 = random.sample(all_choices, 2)
                    if v1 > v2:
                        new_v1 = v2
                        v2 = v1
                        v1 = new_v1
                    predicate = (
                        f"{pred_alias}.{column}>{v1} AND {pred_alias}.{column}<{v2}"
                    )
            else:
                # For other columns, use equality predicates
                value = random.randint(col_min, col_max)
                predicate = f"{pred_alias}.{column}={value}"
            where_conditions.append(predicate)
        # Build the query
        where_clause = " AND ".join(where_conditions)
        query = f"SELECT COUNT(*) FROM {from_clause} WHERE {where_clause}"
        queries.append(query)
    return queries


# Example usage
query_cnt = 4000  # Replace with your desired number of queries
random.seed(0)
queries = generate_queries(query_cnt)

sql_path = "spark-sqls"
name = "job-ext"
with open(f"{sql_path}/{name}_{query_cnt}.txt", "w") as f:
    f.write("\n".join(queries))

os.makedirs(f"{sql_path}/{name}", exist_ok=True)
for qid, q in enumerate(queries):
    with open(f"{sql_path}/{name}/{qid}.sql", "w") as f:
        f.write(q)
