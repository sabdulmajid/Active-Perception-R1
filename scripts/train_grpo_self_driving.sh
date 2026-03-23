#!/usr/bin/env bash
set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

ENGINE="${ENGINE:-vllm}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-VL-3B-Instruct}"
TRAIN_FILE="${TRAIN_FILE:-${REPO_ROOT}/data/self_driving/train.parquet}"
VAL_FILE="${VAL_FILE:-${REPO_ROOT}/data/self_driving/val.parquet}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${REPO_ROOT}/checkpoints/self_driving_grpo}"
PROJECT_NAME="${PROJECT_NAME:-active_perception_r1}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-self_driving_active_vision_grpo}"
REWARD_MODULE="${REWARD_MODULE:-${REPO_ROOT}/src/active_perception_r1/rewards/self_driving_reward.py}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-8}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1536}"
N_RESPONSES="${N_RESPONSES:-4}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}"
ACTOR_MAX_TOKEN_LEN_PER_GPU="${ACTOR_MAX_TOKEN_LEN_PER_GPU:-8192}"
LOGPROB_MAX_TOKEN_LEN_PER_GPU="${LOGPROB_MAX_TOKEN_LEN_PER_GPU:-12288}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.52}"
ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-1}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-5}"
KL_COEF="${KL_COEF:-0.02}"

if [[ -n "${WANDB_API_KEY:-}" ]]; then
  LOGGER='["console","wandb"]'
else
  LOGGER='["console"]'
fi

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=True \
  algorithm.kl_penalty=low_var_kl \
  algorithm.kl_ctrl.type=fixed \
  algorithm.kl_ctrl.kl_coef="${KL_COEF}" \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  data.prompt_key=prompt \
  data.reward_fn_key=data_source \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.val_batch_size="${VAL_BATCH_SIZE}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.image_key=images \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.use_fused_kernels=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
  actor_rollout_ref.actor.optim.weight_decay=0.05 \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.loss_agg_mode=token-mean \
  actor_rollout_ref.actor.grad_clip=1.0 \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${ACTOR_MAX_TOKEN_LEN_PER_GPU}" \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${LOGPROB_MAX_TOKEN_LEN_PER_GPU}" \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.rollout.name="${ENGINE}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP_SIZE}" \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${LOGPROB_MAX_TOKEN_LEN_PER_GPU}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
  actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
  actor_rollout_ref.rollout.max_num_seqs=8 \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.n="${N_RESPONSES}" \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
  reward.custom_reward_function.path="${REWARD_MODULE}" \
  reward.custom_reward_function.name=compute_score \
  +reward.custom_reward_function.reward_kwargs.process_reward_scale=0.35 \
  +reward.custom_reward_function.reward_kwargs.max_zoom_calls=3 \
  +reward.custom_reward_function.reward_kwargs.min_relative_area=0.015 \
  +reward.custom_reward_function.reward_kwargs.max_relative_area=0.55 \
  trainer.use_legacy_worker_impl=disable \
  trainer.logger="${LOGGER}" \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.default_local_dir="${CHECKPOINT_DIR}" \
  trainer.n_gpus_per_node=2 \
  trainer.nnodes=1 \
  trainer.val_before_train=True \
  trainer.critic_warmup=0 \
  trainer.save_freq=10 \
  trainer.test_freq=10 \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  "$@"

