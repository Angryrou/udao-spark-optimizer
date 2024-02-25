for n_samples in 100000
do
  for n_ws in 11 21 51 101
  do
    python compile_time_hierarchical_optimizer.py --benchmark tpch --q_type qs_lqp_compile --graph_choice gtn --use_mlp --moo_algo "ws" --save_data --save_data_header "./output/0221expr/tpch" --n_samples $n_samples --n_ws $n_ws
    python compile_time_hierarchical_optimizer.py --benchmark tpch --q_type qs_lqp_compile --graph_choice gtn --use_mlp --moo_algo "ws" --save_data --save_data_header "./output/0221expr/tpch" --n_samples $n_samples --n_ws $n_ws --set_query_control
  done
done