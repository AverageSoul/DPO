#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
mkdir -p logs
log_path="logs/hh_sft_pythia28_4x4090_$(date +%Y%m%d_%H%M%S).log"
CUDA_VISIBLE_DEVICES=4,5,6,7 HYDRA_FULL_ERROR=1 python -u train.py \
  model=pythia28 \
  model.name_or_path=/mnt/LLM/models/pythia-2.8b \
  model.tokenizer_name_or_path=/mnt/LLM/models/pythia-2.8b \
  model.policy_dtype=float32 \
  model.fsdp_policy_mp=bfloat16 \
  datasets=[hh] \
  dataset_path=/mnt/LLM/datasets/hh-rlhf \
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
  n_eval_examples=1000 \
  eval_every=2000 \
  do_first_eval=true \
  sample_during_eval=false \
  wandb.enabled=false \
  debug=false \
  'local_dirs=[.cache]' \
  exp_name=hh_sft_pythia28_4x4090 \
  "$@" 2>&1 | tee "$log_path"
