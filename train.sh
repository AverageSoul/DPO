#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

# 可通过环境变量修改
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
MODEL_PATH="${MODEL_PATH:-/mnt1/zhouyeying/models/Qwen2.5-0.5B-Instruct}"
DATASET_PATH="${DATASET_PATH:-/mnt1/zhouyeying/datasets/hh-rlhf}"

mkdir -p logs
log_path="logs/hh_sft_qwen25_05b_instruct_4x3090_$(date +%Y%m%d_%H%M%S).log"

HYDRA_FULL_ERROR=1 python -u train.py \
  model=pythia28 \
  "model.name_or_path=$MODEL_PATH" \
  "model.tokenizer_name_or_path=$MODEL_PATH" \
  model.block_name=Qwen2DecoderLayer \
  model.policy_dtype=float32 \
  model.fsdp_policy_mp=bfloat16 \
  datasets=[hh] \
  "dataset_path=$DATASET_PATH" \
  loss=sft \
  trainer=FSDPTrainer \
  batch_size=16 \
  eval_batch_size=4 \
  gradient_accumulation_steps=4 \
  max_length=512 \
  max_prompt_length=256 \
  activation_checkpointing=true \
  optimizer=RMSprop \
  lr=5e-7 \
  n_epochs=1 \
  n_eval_examples=256 \
  eval_every=2000 \
  do_first_eval=true \
  sample_during_eval=false \
  wandb.enabled=false \
  debug=false \
  'local_dirs=[.cache]' \
  exp_name=hh_sft_qwen25_05b_instruct_4x3090 \
  "$@" 2>&1 | tee "$log_path"k
