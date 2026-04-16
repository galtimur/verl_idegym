set -x

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3,5}"

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"

CONFIG_PATH="${PROJECT_DIR}/examples/django_idegym/config"

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='django_idegym_grpo' \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=8 \
    data.train_files=JetBrains-Research/django_method_gen:train \
    data.val_files=JetBrains-Research/django_method_gen:test \
    data.custom_cls.path="${PROJECT_DIR}/examples/django_idegym/hf_dataset.py" \
    data.custom_cls.name=HFHubDataset \
    actor_rollout_ref.model.path=Qwen/Qwen3-0.6B \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=null \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="${PROJECT_DIR}/examples/django_idegym/config/agent_loop_config.yaml" \
    trainer.total_epochs=15 \
    trainer.logger=['wandb'] \
    trainer.project_name=django_idegym \
    trainer.experiment_name=django_idegym_grpo \
    trainer.n_gpus_per_node=4 \
    "$@"
