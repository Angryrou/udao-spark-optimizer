for n_samples in 100000 1000000 10000000 100000000
do
  for n_ws in 11
  do
    python compile_time_hierarchical_optimizer1.py --q_type qs_lqp_compile --ag_model_qs_ana_latency "WeightedEnsemble_L2_FULL" --ag_model_qs_io "CatBoost" --graph_choice gtn --infer_limit 1e-5 --infer_limit_batch_size 10000 --ag_time_limit 21600 --ag_sign "high_quality" --moo_algo "ws" --save_data --save_data_header "./output/0218test/smaller_range/" --n_samples $n_samples --n_ws $n_ws
    python compile_time_hierarchical_optimizer1.py --q_type qs_lqp_compile --ag_model_qs_ana_latency "WeightedEnsemble_L2_FULL" --ag_model_qs_io "CatBoost" --graph_choice gtn --infer_limit 1e-5 --infer_limit_batch_size 10000 --ag_time_limit 21600 --ag_sign "high_quality" --moo_algo "ws" --save_data --save_data_header "./output/0218test/smaller_range/" --n_samples $n_samples --n_ws $n_ws --set_query_control
  done
done