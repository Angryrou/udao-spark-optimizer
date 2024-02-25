for n_c in 16 32 64 128 256
do
  for n_p in 16 32 64 128 256 512
  do
    python compile_time_hierarchical_optimizer.py --q_type qs_lqp_compile --ag_model_qs_ana_latency "WeightedEnsemble_L2_FULL" --ag_model_qs_io "CatBoost" --graph_choice gtn --infer_limit 1e-5 --infer_limit_batch_size 10000 --ag_time_limit 21600 --ag_sign "high_quality" --moo_algo "div_and_conq_moo%B" --sample_mode "grid" --save_data --save_data_header "./output/0218test/" --n_c_samples $n_c --n_p_samples $n_p
    python compile_time_hierarchical_optimizer.py --q_type qs_lqp_compile --ag_model_qs_ana_latency "WeightedEnsemble_L2_FULL" --ag_model_qs_io "CatBoost" --graph_choice gtn --infer_limit 1e-5 --infer_limit_batch_size 10000 --ag_time_limit 21600 --ag_sign "high_quality" --moo_algo "div_and_conq_moo%B" --sample_mode "random" --save_data --save_data_header "./output/0218test/" --n_c_samples $n_c --n_p_samples $n_p
  done
done