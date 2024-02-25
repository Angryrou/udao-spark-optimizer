for n_c in 64
do
  for n_p in 128
  do
    python compile_time_hierarchical_optimizer.py --benchmark tpch --q_type qs_lqp_compile --graph_choice gtn --use_mlp --moo_algo "div_and_conq_moo%GD" --sample_mode "grid" --save_data --save_data_header "./output/0221expr/tpch" --n_c_samples $n_c --n_p_samples $n_p
    python compile_time_hierarchical_optimizer.py --benchmark tpch --q_type qs_lqp_compile --graph_choice gtn --use_mlp --moo_algo "div_and_conq_moo%WS&11" --sample_mode "grid" --save_data --save_data_header "./output/0221expr/tpch" --n_c_samples $n_c --n_p_samples $n_p
  done
done