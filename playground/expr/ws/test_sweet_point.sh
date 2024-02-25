for n_samples in 100000
do
  for n_ws in 11 21 51 101
  do
    python compile_time_hierarchical_optimizer.py --q_type qs_lqp_compile --ag_model_qs_ana_latency "WeightedEnsemble_L2_FULL" --ag_model_qs_io "CatBoost" --graph_choice gtn --infer_limit 1e-5 --infer_limit_batch_size 10000 --ag_time_limit 21600 --ag_sign "high_quality" --moo_algo "ws" --save_data --save_data_header "./output/0218test/n_ws_test" --n_samples $n_samples --n_ws $n_ws
    python compile_time_hierarchical_optimizer.py --q_type qs_lqp_compile --ag_model_qs_ana_latency "WeightedEnsemble_L2_FULL" --ag_model_qs_io "CatBoost" --graph_choice gtn --infer_limit 1e-5 --infer_limit_batch_size 10000 --ag_time_limit 21600 --ag_sign "high_quality" --moo_algo "ws" --save_data --save_data_header "./output/0218test/n_ws_test" --n_samples $n_samples --n_ws $n_ws --set_query_control
  done
done