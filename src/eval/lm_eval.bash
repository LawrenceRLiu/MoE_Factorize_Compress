#!/usr/bin/env bash
# This script is used to evaluate a language model on various tasks.
export CUDA_LAUNCH_BLOCKING=1


#expect to have two inputs
#first input is the model path
#second input is the save path
mkdir -p ${2}/${1}/base

lm-eval --model hf \
    --model_args "pretrained=$1" \
    --tasks leaderboard_gpqa_diamond \
    --log_samples \
    --num_fewshot 0 \
    --batch_size auto \
    --output_path ${2}/${1}/base/gpqa_diamond.json
