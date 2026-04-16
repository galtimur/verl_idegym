# Django IDEGym SweMini RL Training

Multi-turn reinforcement learning for Django method generation using an agentic SWE-style workflow with IDEGym.

## Overview

This example implements a **SweMini-style agent loop** that:
1. Allocates a persistent IDEGym server and cuts the target method
2. Runs an agentic loop: model generates bash commands → commands execute on server → observations returned
3. Supports two modes: **toolcall** (LLM tool_calls) and **text** (```mswea_bash_command blocks)
4. Agent explicitly submits via `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`
5. Tests run only after submission, then rewards are computed

## Architecture

```
initialize → agent_step (loop) → run_tests → finalize → END
              ↑___________|
               (if should_continue)
```

Each `agent_step`:
1. Calls the LLM with conversation history
2. Parses bash commands from response (toolcall or text mode)
3. Executes commands on IDEGym server
4. Renders observations back to conversation

## Extra Dependencies

```bash
pip install langchain-core langgraph aiohttp pyyaml jinja2 minisweagent
# For real IDEGym (not mock):
pip install idegym kubernetes-asyncio
# Optional, for S3 trajectory logging:
pip install s3fs
```

## Directory Structure

```
examples/django_idegym/
├── agent_loop/
│   ├── django_agent_loop.py           # SweMini agent loop + LangGraph nodes
│   ├── chat_model.py                  # LangChain ChatModel wrapper for verl
│   ├── agent_parsing_strategies.py    # Toolcall/Text parsing strategies
│   ├── idegym_runner.py               # IDEGym cloud runner
│   └── mock_idegym_runner.py          # Mock runner for testing
├── reward/
│   ├── django_reward.py               # Reward computation + standalone IDEGym client
│   └── idegym_runner_utils.py         # ItemToRun dataclass
├── utils/
│   ├── postprocessing.py              # Code normalization, test parsing
│   └── reward_helper_fns.py           # Code block extraction, reasoning filter
├── prompts/
│   ├── swemini_prompts.py             # Prompt loader
│   ├── prompts_swemini_toolcall.yaml  # Toolcall mode prompts
│   └── prompts_swemini_text.yaml      # Text mode prompts
└── config/
    ├── django_idegym_grpo.yaml        # Training config
    └── agent_loop_config.yaml         # Agent loop registration
```

## IDEGym Setup

Set environment variables for IDEGym access:
```bash
export ORCHESTRATOR_URL="http://idegym-orchestrator.idegym.svc.cluster.local"
export IMAGE_TAG="your-django-image-tag"
export IDEGYM_AUTH_USERNAME="your-username"
export IDEGYM_AUTH_PASSWORD="your-password"
```

For testing without IDEGym, set `use_mock_runner: true` in the config.

## Usage

```bash
bash examples/django_idegym/run_django_idegym.sh
```

With mock runner (no IDEGym needed):
```bash
bash examples/django_idegym/run_django_idegym.sh \
    actor_rollout_ref.rollout.use_mock_runner=true
```

## Key Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rollout.max_turns` | 10 | Max agent interaction steps |
| `rollout.agent_parsing_mode` | "toolcall" | "toolcall" or "text" |
| `rollout.use_mock_runner` | false | Use mock IDEGym runner |
| `rollout.max_num_tests` | 10 | Max tests per item |
| `rollout.max_test_output_symb` | 10000 | Max chars of output in observations |
| `rollout.keep_reasoning` | "none" | Filter `<think>` blocks: "none", "last", "all" |
| `rollout.enable_thinking` | false | Enable extended thinking in tokenizer |
| `rollout.trajectory_dir` | null | S3 path for trajectory saving |
