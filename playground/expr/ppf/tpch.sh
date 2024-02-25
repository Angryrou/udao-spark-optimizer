for n_grids in 1 2 4
do
  for n_max_iters in 1 2 4
  do
    python compile_time_hierarchical_optimizer.py --benchmark tpch --q_type qs_lqp_compile --graph_choice gtn --use_mlp --moo_algo "ppf" --save_data --save_data_header "./output/0221expr/tpch" --n_process 1 --n_grids $n_grids --n_max_iters $n_max_iters
    python compile_time_hierarchical_optimizer.py --benchmark tpch --q_type qs_lqp_compile --graph_choice gtn --use_mlp --moo_algo "ppf" --save_data --save_data_header "./output/0221expr/tpch" --n_process 1 --n_grids $n_grids --n_max_iters $n_max_iters --set_query_control
  done
done