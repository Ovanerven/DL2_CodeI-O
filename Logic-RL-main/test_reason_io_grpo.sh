#!/bin/bash
set -x

# Replace with your model path - use a smaller model for faster testing
MODEL_PATH="Qwen/Qwen2.5-1.5B-Instruct"  # Smaller model for testing

# Export environment variable for XFormers attention backend (optional)
export VLLM_ATTENTION_BACKEND=XFORMERS

# Specify the test dataset path
TEST_DATASET="../test_logicrl_dataset_preview.parquet"

# Check if dataset exists
if [ ! -f "$TEST_DATASET" ]; then
    echo "Test dataset not found at $TEST_DATASET"
    echo "Generating test dataset..."
    cd ..
    python CodeIO-RL/reasoning_questions.py --preview --logic_rl_format --output test_logicrl_dataset_preview.parquet
    cd Logic-RL-main
fi

# Run a minimal PPO training iteration for testing
python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$TEST_DATASET \
    data.val_files=$TEST_DATASET \
    data.train_batch_size=2 \
    data.val_batch_size=1 \
    data.max_prompt_length=400 \
    data.max_response_length=1024 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=3e-7 \
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
    actor_rollout_ref.rollout.log_prob_micro_batch_size=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=[] \
    trainer.project_name='GRPO_ReasonIO_Test' \
    trainer.experiment_name='ReasonIO-Test' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.default_local_dir=./test_outputs \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=1 \
    trainer.test_freq=1 \
    trainer.total_epochs=1 $@ 2>&1 | tee test_reason_io_grpo.log 