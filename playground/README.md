NOTE: under the directory: `playground`

## For tpch: 

- DivB (Original)
```python
save_data_header = f"./output/0218test"
cpu_mode = "cpu"
sample_mode = "grid"
is_oracle = False
model_name = "ag"
div_moo_setting = "128_256"
method = "div_and_conq_moo%B"
dag_opt_algo = "B"
# query_id: the id of query, e.g. 2-1
# n_s: the number of subQs
data_path = f"{save_data_header}/query_control_False/latest_model_{cpu_mode}/{model_name}/oracle_False/{method}/" \
                        f"{div_moo_setting}/time_-1/query_{query_id}_n_{n_s}/{sample_mode}/{dag_opt_algo}"
```

- DivB (More-resources-setting-with-failure-classifier)
```python
save_data_header = "./output/0218test/tpch_traces/new_test_end"
cpu_mode = "cpu"
sample_mode = "grid"
is_oracle = False
model_name = "ag"
div_moo_setting = "256_256"
method = "div_and_conq_moo%B"
dag_opt_algo = "B"
# query_id: the id of query, e.g. 2-1
# n_s: the number of subQs
data_path = f"{save_data_header}/query_control_False/latest_model_{cpu_mode}/{model_name}/oracle_False/{method}/" \
                        f"{div_moo_setting}/time_-1/query_{query_id}_n_{n_s}/{sample_mode}/{dag_opt_algo}"
```

- DivB (Oracle-setting-with-failure-classifier)
```python
save_data_header = "./output/0218test/oracle_True/tpch_traces/new_test_end"
cpu_mode = "cpu"
sample_mode = "grid"
is_oracle = True
model_name = "ag"
div_moo_setting = "256_256"
method = "div_and_conq_moo%B"
dag_opt_algo = "B"
# query_id: the id of query, e.g. 2-1
# n_s: the number of subQs
data_path = f"{save_data_header}/query_control_False/latest_model_{cpu_mode}/{model_name}/oracle_True/{method}/" \
                        f"{div_moo_setting}/time_-1/query_{query_id}_n_{n_s}/{sample_mode}/{dag_opt_algo}"
```

- WS (query-level, without failure-classifier):

```python
save_data_header = "./output/0218test"
cpu_mode = "cpu"
model_name = "ag"
ws_setting = "100000_11"
method = "ws"
# query_id: the id of query, e.g. 2-1
# n_s: the number of subQs
data_path = f"{save_data_header}/query_control_True/latest_model_{cpu_mode}/{model_name}/oracle_False/{method}/" \
                        f"{ws_setting}/time_-1/query_{query_id}_n_{n_s}"
```

- WS (query-level, with failure-classifier):

```python
save_data_header = "./output/0218test/tpch_traces/rerun_fail_queries"
cpu_mode = "cpu"
model_name = "ag"
ws_setting = "100000_11"
method = "ws"
# query_id: the id of query, e.g. 2-1
# n_s: the number of subQs
# only rerun query 4-1, 7-1 and 13-1
data_path = f"{save_data_header}/query_control_True/latest_model_{cpu_mode}/{model_name}/oracle_False/{method}/" \
                        f"{ws_setting}/time_-1/query_{query_id}_n_{n_s}"
```

- EVO (query-level, without failure-classifier):

```python
save_data_header = "./output/0218test"
cpu_mode = "cpu"
model_name = "ag"
evo_setting = "100_500"
method = "evo"
# query_id: the id of query, e.g. 2-1
# n_s: the number of subQs
data_path = f"{save_data_header}/query_control_True/latest_model_{cpu_mode}/{model_name}/oracle_False/{method}/" \
                        f"{evo_setting}/time_-1/query_{query_id}_n_{n_s}"
```

- PF (query-level, without failure-classifier):

```python
save_data_header = "./output/0218test"
cpu_mode = "cpu"
model_name = "ag"
ppf_setting = "1_4_4"
method = "ppf"
# query_id: the id of query, e.g. 2-1
# n_s: the number of subQs
data_path = f"{save_data_header}/query_control_True/latest_model_{cpu_mode}/{model_name}/oracle_False/{method}/" \
                        f"{ppf_setting}/time_-1/query_{query_id}_n_{n_s}"
```

## For tpcds: 

- DivB (More-resources-setting-with-failure-classifier)
```python
save_data_header = "./output/0218test/tpcds_traces/new_test_end"
cpu_mode = "cpu"
sample_mode = "grid"
model_name = "ag"
div_moo_setting = "512_32"
method = "div_and_conq_moo%B"
dag_opt_algo = "B"
# query_id: the id of query, e.g. 2-1
# n_s: the number of subQs
data_path = f"{save_data_header}/query_control_False/latest_model_{cpu_mode}/{model_name}/oracle_False/{method}/" \
                        f"{div_moo_setting}/time_-1/query_{query_id}_n_{n_s}/{sample_mode}/{dag_opt_algo}"
```

- DivB (Oracle-setting-with-failure-classifier)
```python
save_data_header = "./output/0218test/oracle_True/tpcds_traces/new_test_end"
cpu_mode = "cpu"
sample_mode = "grid"
is_oracle = True
model_name = "ag"
div_moo_setting = "512_32"
method = "div_and_conq_moo%B"
dag_opt_algo = "B"
# query_id: the id of query, e.g. 2-1
# n_s: the number of subQs
data_path = f"{save_data_header}/query_control_False/latest_model_{cpu_mode}/{model_name}/oracle_True/{method}/" \
                        f"{div_moo_setting}/time_-1/query_{query_id}_n_{n_s}/{sample_mode}/{dag_opt_algo}"
```

- WS (query-level, without failure-classifier):

```python
save_data_header = "./output/0218test/tpcds_traces"
cpu_mode = "cpu"
model_name = "ag"
ws_setting = "100000_11"
method = "ws"
# query_id: the id of query, e.g. 2-1
# n_s: the number of subQs
data_path = f"{save_data_header}/query_control_True/latest_model_{cpu_mode}/{model_name}/oracle_False/{method}/" \
                        f"{ws_setting}/time_-1/query_{query_id}_n_{n_s}"
```

- EVO (query-level, without failure-classifier):

```python
save_data_header = "./output/0218test/tpcds_traces"
cpu_mode = "cpu"
model_name = "ag"
evo_setting = "100_500"
method = "evo"
# query_id: the id of query, e.g. 2-1
# n_s: the number of subQs
data_path = f"{save_data_header}/query_control_True/latest_model_{cpu_mode}/{model_name}/oracle_False/{method}/" \
                        f"{evo_setting}/time_-1/query_{query_id}_n_{n_s}"
```


- PF (query-level, without failure-classifier):

```python
save_data_header = "./output/0218test/tpcds_traces"
cpu_mode = "cpu"
model_name = "ag"
ws_setting = "1_2_2"
method = "ppf"
# query_id: the id of query, e.g. 2-1
# n_s: the number of subQs
data_path = f"{save_data_header}/query_control_True/latest_model_{cpu_mode}/{model_name}/oracle_False/{method}/" \
                        f"{ppf_setting}/time_-1/query_{query_id}_n_{n_s}"
```



