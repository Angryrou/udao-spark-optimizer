for n_grids in 1 2 4
do
  for n_max_iters in 1 2 4
  do
    python compile_time_hierarchical_optimizer.py --q_type qs_lqp_compile --ag_model_qs_ana_latency "WeightedEnsemble_L2_FULL" --ag_model_qs_io "CatBoost" --graph_choice gtn --infer_limit 1e-5 --infer_limit_batch_size 10000 --ag_time_limit 21600 --ag_sign "high_quality" --moo_algo "ppf" --save_data --save_data_header "./output/0218test/" --n_process 1 --n_grids $n_grids --n_max_iters $n_max_iters
    python compile_time_hierarchical_optimizer.py --q_type qs_lqp_compile --ag_model_qs_ana_latency "WeightedEnsemble_L2_FULL" --ag_model_qs_io "CatBoost" --graph_choice gtn --infer_limit 1e-5 --infer_limit_batch_size 10000 --ag_time_limit 21600 --ag_sign "high_quality" --moo_algo "ppf" --save_data --save_data_header "./output/0218test/" --n_process 1 --n_grids $n_grids --n_max_iters $n_max_iters --set_query_control
  done
done