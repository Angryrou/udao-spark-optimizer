for n_c in 16 32 64 128 256
do
  for n_p in 16 32 64 128 256 512
  do
    python compile_time_hierarchical_optimizer.py --q_type qs_lqp_compile --ag_model "WeightedEnsemble_L2" --graph_choice gtn --infer_limit 1e-5 --infer_limit_batch_size 10000 --hp_choice tuned-0202 --moo_algo "div_and_conq_moo%B" --sample_mode "grid" --save_data --n_c_samples $n_c --n_p_samples $n_p
    python compile_time_hierarchical_optimizer.py --q_type qs_lqp_compile --ag_model "WeightedEnsemble_L2" --graph_choice gtn --infer_limit 1e-5 --infer_limit_batch_size 10000 --hp_choice tuned-0202 --moo_algo "div_and_conq_moo%B" --sample_mode "random" --save_data --n_c_samples $n_c --n_p_samples $n_p
  done
done