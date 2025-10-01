#!/bin/bash

#qwen compression 
enviroment="ARMOR_main"

source $(conda info --base)/etc/profile.d/conda.sh
conda activate $enviroment


model=$1
method=$2
#third args are optional
if [ -z "$3" ]; then
    num_processes=-1
    parallelize=false
else
    num_processes=$3
    parallelize=true
fi


gpus=4,5,6,7 #change this for your machine
pattern="2_4" #choices are 2_4, 4_8, unstructured
seqlen=8192

cd "wanda-main"

run_name="${method}/${pattern}"

cmd="CUDA_VISIBLE_DEVICES=${gpus} python -u main.py\
    --model=$model\
    --sparsity_ratio=0.5\
    --prune_method=$method\
    --seqlen=8192\
    --save_model=../models/${model}/compressed/${run_name}/model"

if [ "$pattern" = "unstructured" ]; then
    cmd+=" --sparsity_type=unstructured"
#if the pattern is 2_4
elif [ "$pattern" = "2_4" ]; then
    cmd+=" --sparsity_type=2:4"
#if the pattern is 4_8
elif [ "$pattern" = "4_8" ]; then
    cmd+=" --sparsity_type=4:8"
#otherwise raise an error
else
    echo "Unknown pattern: $pattern"
    exit 1

fi

echo "Running command: $cmd"

eval $cmd
if [ $? -ne 0 ]; then
    echo "Command failed, check the log file"
    exit 1
fi

scripts/evaluation/pretrain_evaluation.bash \
    model_path="./models/${model}/compressed/${run_name}/model" \
    model_name="$model" \
    generate_non_compressed_model=false \
    results_path="./models/${model}/compressed/${run_name}/eval" \
    gpus="$gpus" \
    parallelize=$parallelize \
    num_processes=$num_processes
