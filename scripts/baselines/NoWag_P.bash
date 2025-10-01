#!/bin/bash

enviroment="ARMOR_main"

source $(conda info --base)/etc/profile.d/conda.sh
conda activate $enviroment

#defualt args

model=$1
#third args are optional
if [ -z "$2" ]; then
    num_processes=-1 #for inference stuff
    parallelize=false
else
    num_processes=$2
    parallelize=true
fi


gpus=4,5,6,7 #change this for your machine

pattern="2_4" 
seqlen=8192

#loop through the args
for arg in "$@"
do
    # Check if the argument is in key=value format
    if [[ "$arg" == *"="* ]]; then
        # Split into key and value
        key="${arg%%=*}"
        value="${arg#*=}"

        # 3. Use 'declare' again to update the variable
        # This will only affect variables already declared above.
        # For safety, you can check if the variable exists first.
        if -p "$key" &>/dev/null; then
            "$key=$value"
            echo "Updated argument '$key' to '$value'"
        else
            echo "ERROR: Unknown argument '$key'"
            exit 1
        fi
    fi
done

run_name="NoWag_P/${pattern}"


generate_cmd="CUDA_VISIBLE_DEVICES=${gpus} scripts/calibration_data_generation/generate.bash ${model} SlimPajama-627B 128 ${seqlen}"
echo "$generate_cmd"
eval "$generate_cmd"

cmd="CUDA_VISIBLE_DEVICES=${gpus} python -u ParallelCompress.py \
    base_model=$model \
    compress=prune \
    run_name=$run_name \
    compress.compression_config.normalizer_kwargs.norm_order=[0,1] \
    \"datasets=[{dataset_config:SlimPajama-627B,n_samples:128,ctx_len:${seqlen}}]\""

if [ "$pattern" == "2_4" ]; then
    cmd+=" +compress.compression_config.pattern=[2,4]"
elif [ "$pattern" == "4_8" ]; then
    cmd+=" +compress.compression_config.pattern=[4,8]"
elif [ "$pattern" == "unstructured" ]; then
    continue #do nothing, unstructured pruning is the default
else
    echo "ERROR: Unknown pattern '$pattern'"
    exit 1
fi

echo "Command to run: $cmd"


results_path="models/${model}/compressed/$run_name"
    
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $enviroment
eval $cmd
if [ $? -ne 0 ]; then
    echo "Compression failed for model: $model"
else
    # rm -rf "../LLM_data"


    #evaluate
    log_dir_use="./logs/${model}/"
    mkdir -p "$log_dir_use"
    echo "Evaluating model: $model"

    scripts/evaluation/pretrain_evaluation.bash \
        model_path="${results_path}" \
        model_name="$model" \
        generate_non_compressed_model=true \
        results_path="${results_path}/eval" \
        evaluate_perplexity=false \
        parallelize=$parallelize \
        num_processes=$num_processes \
        max_length=4096 \
        gpus="$gpus" > "${log_dir_use}/NoWagP_${pattern}.log" 2>&1
done
