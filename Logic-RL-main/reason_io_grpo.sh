#!/bin/bash
set -x

# Replace with your model path
MODEL_PATH=YOUR_MODEL_PATH  # e.g., "Qwen/Qwen2.5-7B-Instruct"

# Export environment variable for XFormers attention backend (optional, but can improve performance)
export VLLM_ATTENTION_BACKEND=XFORMERS

# Ensure the data directory exists
mkdir -p Logic-RL-main/data/reason_io

# Run the conversion script if needed
python CodeIO-RL/convert_to_logicrl_format.py

# Run the PPO training
python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=Logic-RL-main/data/reason_io/dataset.parquet \
    data.val_files=Logic-RL-main/data/reason_io/dataset.parquet \
    data.train_batch_size=4 \
    data.val_batch_size=2 \
    data.max_prompt_length=400 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=3e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.actor.ppo_micro_batch_size=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.grad_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb'] \
    trainer.project_name='GRPO_ReasonIO' \
    trainer.experiment_name='ReasonIO-RL' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.default_local_dir=./outputs \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 $@ 2>&1 | tee reason_io_grpo.log 