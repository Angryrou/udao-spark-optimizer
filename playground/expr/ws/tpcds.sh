for n_samples in 100000
do
  for n_ws in 11 21 51 101
  do
    python compile_time_hierarchical_optimizer.py --benchmark tpcds --q_type qs_lqp_compile --graph_choice gtn --infer_limit 1e-5 --infer_limit_batch_size 10000 --use_mlp --moo_algo "ws" --save_data --save_data_header "./output/0221expr/tpcds" --n_samples $n_samples --n_ws $n_ws
    python compile_time_hierarchical_optimizer.py --benchmark tpcds --q_type qs_lqp_compile --graph_choice gtn --infer_limit 1e-5 --infer_limit_batch_size 10000 --use_mlp --moo_algo "ws" --save_data --save_data_header "./output/0221expr/tpcds" --n_samples $n_samples --n_ws $n_ws --set_query_control
  done
done