for n_c in 64
do
  for n_p in 256
  do
    python compile_time_hierarchical_optimizer.py --benchmark tpcds --q_type qs_lqp_compile --graph_choice gtn --infer_limit 1e-5 --infer_limit_batch_size 10000 --use_mlp --moo_algo "div_and_conq_moo%GD" --sample_mode "grid" --save_data --save_data_header "./output/0221expr/tpcds" --n_c_samples $n_c --n_p_samples $n_p
    python compile_time_hierarchical_optimizer.py --benchmark tpcds --q_type qs_lqp_compile --graph_choice gtn --infer_limit 1e-5 --infer_limit_batch_size 10000 --use_mlp --moo_algo "div_and_conq_moo%WS&11" --sample_mode "grid" --save_data --save_data_header "./output/0221expr/tpcds" --n_c_samples $n_c --n_p_samples $n_p
  done
done