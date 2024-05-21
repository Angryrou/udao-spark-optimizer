bm=$1
cname=$2
weights=$3
nreps=$4
runtime=$5

if [ "$bm" = "tpch" ]; then
  host=node21-opa
  bt=TPCH
  cn=HEX1
elif [ "$bm" = "tpcds" ]; then
  host=localhost
  bt=TPCDS
  cn=HEX2
else
  echo "Invalid benchmark specified"
  exit 1
fi

if [ "$runtime" -eq 1 ]; then
    python spark_trace_on_demand.py \
    --trace_header "evaluations" \
    --n_data_per_template 1 \
    --benchmark_type $bt \
    --cluster_name $cn \
    --n_processes 1 \
    --cluster_cores 120 \
    --n_reps $nreps \
    --enable_runtime_optimizer \
    --runtime_optimizer_host $host \
    --runtime_optimizer_port 12345 \
    --configuration_header divB_new_grids/on_demand \
    --configuration_name ${cname}_${weights}.json
else
    python spark_trace_on_demand.py \
    --trace_header "evaluations" \
    --n_data_per_template 1 \
    --benchmark_type $bt \
    --cluster_name $cn \
    --n_processes 16 \
    --cluster_cores 120 \
    --n_reps $nreps \
    --configuration_header divB_new_grids/on_demand \
    --configuration_name ${cname}_${weights}.json
fi
