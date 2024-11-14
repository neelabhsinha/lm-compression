#!/bin/bash
#SBATCH -JLM_Compression
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --gres=gpu:H100:1
#SBATCH --mem-per-gpu=240GB
#SBATCH -t3:00:00
#SBATCH -ologs/Report-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nsinha68@gatech.edu


export PYTHONUNBUFFERED=TRUE
source ~/.bashrc
conda activate sysml_env
cd ~/scratch/lm-compression

model_name="meta-llama/Meta-Llama-3-8B-Instruct"
decoding_strategy="greedy"
initial_local_window=512
steepness_coefficient=1
sink_tokens=4
seq_pooling_type="mean"
compress_context=false
batch_size=4
dataset_split="test"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --model_name) model_name="$2"; shift 2 ;;
    --decoding_strategy) decoding_strategy="$2"; shift 2 ;;
    --initial_local_window) initial_local_window="$2"; shift 2 ;;
    --steepness_coefficient) steepness_coefficient="$2"; shift 2 ;;
    --sink_tokens) sink_tokens="$2"; shift 2 ;;
    --seq_pooling_type) seq_pooling_type="$2"; shift 2 ;;
    --compress_context) compress_context=true; shift 1 ;;
    --batch_size) batch_size="$2"; shift 2 ;;
    --dataset_split) dataset_split="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

cmd="srun -u python -u main.py --model_name $model_name --decoding_strategy $decoding_strategy --initial_local_window $initial_local_window --steepness_coefficient $steepness_coefficient --sink_tokens $sink_tokens --seq_pooling_type $seq_pooling_type --batch_size $batch_size --dataset_split $dataset_split"

[[ "$compress_context" = true ]] && cmd+=" --compress_context"

eval $cmd