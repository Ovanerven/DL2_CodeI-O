#!/bin/bash
set -x

# Replace with your model path
MODEL_PATH=Qwen/Qwen2.5-3B-Instruct

# Export environment variable for XFormers attention backend (optional, but can improve performance)
export VLLM_ATTENTION_BACKEND=XFORMERS

timestamp=$(date +"%Y-%m-%d_%H-%M-%S")

# Ensure the data directory exists
mkdir -p data/reason_io
# Ensure the logs directory exists
mkdir -p logs/reasonio

# Define the system prompt (with double escaping for JSON in bash)
SYSTEM_PROMPT="You are a helpful assistant. The assistant first thinks about the reasoning process and then provides the user with the answer. The reasoning process should be enclosed within <think> </think> tags, i.e., <think> reasoning process here </think>. For your final answer, you must format it as a JSON object, exactly as specified in the prompt, and enclose it within <answer> </answer> tags. For example: <answer>{\\\"output\\\": value}</answer> or <answer>{\\\"input\\\": value}</answer> depending on what's requested. When formatting your JSON output, never use Python expressions like 10**10 (use 10000000000 instead). Now the user asks you to solve a complex problem. After thinking through your reasoning, clearly state your answer as a properly formatted JSON object within answer tags."

# Print the system prompt for verification
echo "System prompt:"
echo "$SYSTEM_PROMPT"

# Run the PPO training with memory-optimized settings
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=reinforce_plus_plus \
    data.train_files=data/reason_io/reason_io_dataset_preview.parquet \
    data.val_files=data/reason_io/reason_io_dataset_preview.parquet \
    data.train_batch_size=1 \
    data.val_batch_size=1 \
    data.max_prompt_length=10000 \
    data.max_response_length=4096 \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=4e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=1 \
    actor_rollout_ref.actor.ppo_micro_batch_size=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.grad_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=160 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=160 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb'] \
    trainer.project_name='ReinforcePP_ReasonIO' \
    trainer.experiment_name="Qwen-3B_ReasonIO_${timestamp}" \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.default_local_dir=actor_checkpoints/reason_io_${timestamp} \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=500 \
    trainer.test_freq=10 \
    trainer.total_epochs=5 $@ 2>&1 | tee logs/reasonio/reason_io_${timestamp}.log
    