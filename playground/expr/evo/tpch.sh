for pop_size in 10 20 50 100
do
  for nfe in 100 200 500
  do
    python compile_time_hierarchical_optimizer.py --benchmark tpch --q_type qs_lqp_compile --graph_choice gtn --use_mlp --moo_algo "evo" --save_data --save_data_header "./output/0221expr/tpch" --pop_size $pop_size --nfe $nfe
    python compile_time_hierarchical_optimizer.py --benchmark tpch --q_type qs_lqp_compile --graph_choice gtn --use_mlp --moo_algo "evo"  --save_data --save_data_header "./output/0221expr/tpch" --pop_size $pop_size --nfe $nfe --set_query_control
  done
done