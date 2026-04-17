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

## Installation

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[vllm]"
uv pip install idegym-{tools,api,backend-utils,client,common-utils,rewards,image-builder}==0.8.0
uv pip install langchain-core langgraph aiohttp pyyaml pytest "datasets>=3.6"
uv pip install flash-attn --no-build-isolation
```

For testing without IDEGym, set `use_mock_runner: true` in the config.

To test IDEgym:

`pytest examples/django_idegym/test/test_idegym_runner_integration.py::test_idegym_smoke -v -s`

## Usage

```bash
./examples/django_idegym/run_django_idegym.sh
```

With mock runner (no IDEGym needed):
```bash
./examples/django_idegym/run_django_idegym.sh \
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
