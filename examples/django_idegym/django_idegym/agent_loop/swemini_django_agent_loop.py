"""
Django IDEGym SweMini agent loop for verl.

Implements a LangGraph-based agentic workflow:
  initialize → agent_step (loop) → run_tests → finalize → END

The model generates bash commands executed on a persistent IDEGym server.
After the agent submits (or max turns), tests are run and rewards computed.

Ported from jetrl_django_idegym/scaffold/django_swemini_agent.py.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import time
import traceback
import uuid
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from datetime import datetime
from functools import partial
from typing import Annotated, Any, Sequence, TypedDict

from jinja2 import Environment as JinjaEnvironment
from jinja2 import StrictUndefined
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, convert_to_openai_messages
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from examples.django_idegym.agent_loop.agent_parsing_strategies import FormatError
from openai import BadRequestError

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register

from examples.django_idegym.agent_loop.agent_parsing_strategies import (
    ParsingStrategy,
    TextStrategy,
    ToolcallStrategy,
)
from examples.django_idegym.agent_loop.chat_model import ChatModel, MaxTokenExceededError, convert_to_agent_output
from examples.django_idegym.prompts.swemini_prompts import load_prompts
from examples.django_idegym.reward.idegym_runner_utils import ItemToRun
from examples.django_idegym.utils.postprocessing import (
    get_percentage_passed,
    parse_idegym_tests_output,
)
from examples.django_idegym.utils.reward_helper_fns import apply_reasoning_filter

logger = logging.getLogger(__name__)

SUBMIT_SIGNAL = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
FAILED_PIPELINE_SCORE = 0

KEEP_REASONING_OPTIONS = {"none", "last", "all"}

_CONTAINER_SYSTEM_INFO = {
    "system": "Linux",
    "release": "5.15.0-69-generic",
    "version": "#76-Ubuntu SMP Fri Mar 17 17:19:29 UTC 2023",
    "machine": "x86_64",
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def dataclass_to_dict(obj: Any) -> Any:
    """Convert dataclass objects / BaseMessages to dictionaries recursively."""
    if isinstance(obj, BaseMessage):
        return {"type": obj.__class__.__name__, "content": str(obj.content)}
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return {key: dataclass_to_dict(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [dataclass_to_dict(item) for item in obj]
    return obj


def count_decorators_above(file_lines: list[str], def_line_idx: int) -> int:
    """Count @decorator lines immediately above a function definition."""
    count = 0
    idx = def_line_idx - 1
    while idx >= 0:
        stripped = file_lines[idx].strip()
        if stripped.startswith("@"):
            count += 1
            idx -= 1
        else:
            break
    return count


def get_fallback_message(message: str, role: str = "ai") -> AIMessage | HumanMessage:
    """Create a placeholder message with dummy response_metadata for failed generations."""
    # prompt_ids must contain both prompt and response tokens;
    # response_mask marks which tokens are the response.
    # Use two dummy tokens so prompt_ids[:len-len(mask)] is non-empty.
    metadata = {
        "request_id": str(uuid.uuid4()),
        "prompt_ids": [0, 0],
        "response_mask": [0],
    }
    if role == "ai":
        return AIMessage(content=message, response_metadata=metadata)
    return HumanMessage(content=message)


def trim_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Remove middle messages when context overflows. Keep first 2 + rest after dropping 2."""
    start = messages[:2]
    end = messages[2:]
    end = end[2:]
    return start + end


# ---------------------------------------------------------------------------
# State TypedDict for LangGraph
# ---------------------------------------------------------------------------


class SweMiniState(TypedDict):
    """State passed between the nodes of the LangGraph."""
    messages: Annotated[Sequence[BaseMessage], "Conversation messages"]
    responses_raw: Annotated[Sequence[BaseMessage | str], "Raw LLM responses"]
    test_results: list[dict]
    test_status: str | None
    generated_code_blocks: list[str]
    turn: Annotated[int, "Current turn number"]
    max_turns: Annotated[int, "Maximum turns allowed"]
    should_continue: bool | None
    stop_reason: str | None
    dp_item: dict | None
    edited_file: list[str]
    no_code_block: bool | None
    test_passed: bool | None
    test_output: str | None
    tests_passed_percentage: float | None
    turn_exit: int | None
    timestamp_start: str | None
    timestamp_end: str | None
    duration: float | None
    turn_tests_durations: list[float] | None
    turn_gen_durations: list[float] | None
    reward_components: dict | None
    rm_score: float | None
    # SweMini-specific fields
    client: Any | None
    server: Any | None
    server_id: str | None
    server_init_failed: float | None
    is_failed_rollout: bool | None
    trajectory: list[dict]
    metadata: dict | None
    submit_mode: str | None
    tools_kwargs: dict | None
    extra_info: dict | None


# ---------------------------------------------------------------------------
# CustomChatModel with enable_thinking support
# ---------------------------------------------------------------------------


class CustomChatModel(ChatModel):
    """ChatModel subclass that supports extended thinking via enable_thinking flag."""

    enable_thinking: bool = False

    def __init__(self, *args, **kwargs):
        enable_thinking = kwargs.pop("enable_thinking", False)
        super().__init__(*args, **kwargs)
        self.enable_thinking = enable_thinking

    async def _preprocess(self, messages: list[BaseMessage], **kwargs: Any) -> tuple[str, list[int], list[int]]:
        assert messages[-1].type in ["human", "tool"], (
            f"Last message must be human or tool, but got {messages[-1].type}"
        )
        loop = asyncio.get_running_loop()

        if messages[-1].type == "human" and (len(messages) == 1 or messages[-2].type != "ai"):
            prompt_ids = await loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    convert_to_openai_messages(messages),
                    tools=kwargs.get("tools"),
                    add_generation_prompt=True,
                    tokenize=True,
                    enable_thinking=self.enable_thinking,
                ),
            )
            return str(uuid.uuid4()), prompt_ids, []

        for i in range(len(messages) - 1, -1, -1):
            if messages[i].type == "ai":
                break
        assert "prompt_ids" in messages[i].response_metadata
        assert "response_mask" in messages[i].response_metadata

        tool_responses = convert_to_openai_messages(messages[i + 1 :])
        tool_response_ids = await loop.run_in_executor(
            None,
            lambda messages=tool_responses: self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True, enable_thinking=self.enable_thinking,
            ),
        )
        tool_response_ids = tool_response_ids[len(kwargs["system_prompt"]) :]

        if len(messages[i].response_metadata["response_mask"]) + len(tool_response_ids) >= self.max_tokens:
            raise MaxTokenExceededError(f"Max response length {self.max_tokens} exceeded")

        request_id = messages[i].response_metadata.pop("request_id")
        prompt_ids = messages[i].response_metadata.pop("prompt_ids")
        response_mask = messages[i].response_metadata.pop("response_mask")
        prompt_ids += tool_response_ids
        response_mask += [0] * len(tool_response_ids)

        return request_id, prompt_ids, response_mask


# ---------------------------------------------------------------------------
# call_model node
# ---------------------------------------------------------------------------


async def call_model(
    state: SweMiniState,
    model: ChatModel,
    sampling_params: dict,
    keep_reasoning: str = "none",
    max_turns: int = 10,
    enable_trim: bool = True,
) -> SweMiniState:
    """Query LLM with current messages. Handles context trimming and errors."""
    start_time = time.perf_counter()
    messages = list(state["messages"])

    dp_item = state.get("dp_item")
    if dp_item is None:
        state["should_continue"] = False
        state["stop_reason"] = "no_dp_item"
        state["is_failed_rollout"] = True
        return state

    item_idx = dp_item.get("idx", "?")

    while True:
        new_message, status, caught_exception = await _run_model(messages, model, sampling_params)
        if status in ["context_too_long", "max_tokens_exceeded"] and enable_trim and len(messages) > 2:
            logger.info(f"Trimming context for item idx={item_idx}, turn={state['turn']}")
            messages = trim_messages(messages)
        else:
            break

    state["turn_gen_durations"].append(time.perf_counter() - start_time)

    if new_message is None:
        state["should_continue"] = False
        state["test_status"] = None
        state["is_failed_rollout"] = True
        responses_raw = list(state["responses_raw"])
        responses_raw.append(str(caught_exception))
        state["responses_raw"] = responses_raw
        fallback = get_fallback_message(f"Generation failed: {caught_exception}", role="ai")
        state["messages"].append(fallback)
        return state

    messages = list(state["messages"])
    responses_raw = list(state["responses_raw"])
    responses_raw.append(deepcopy(new_message))
    state["responses_raw"] = responses_raw

    if keep_reasoning in {"none", "last"}:
        apply_reasoning_filter(new_message, max_turns)

    messages.append(new_message)
    state["messages"] = messages
    state["should_continue"] = True
    return state


async def _run_model(
    messages: Sequence[BaseMessage], model: ChatModel, sampling_params: dict[str, Any]
) -> tuple[AIMessage | None, str, Any]:
    caught_exception = ""
    new_message = None
    try:
        new_message = await model.ainvoke(messages, sampling_params=sampling_params)
        status = "success"
    except MaxTokenExceededError as e:
        caught_exception = e
        print(f"Max response length exceeded: {e}")
        status = "max_tokens_exceeded"
    except BadRequestError as e:
        caught_exception = e
        if "maximum context length" in str(e):
            print(f"Context too long: {e}")
            status = "context_too_long"
        else:
            print(f"Generation error occurred: {e}")
            status = "error"
    except Exception as e:
        caught_exception = e
        print(f"Generation error occurred: {e}")
        status = "error"
    return new_message, status, caught_exception


# ---------------------------------------------------------------------------
# SWEMiniDjangoAgentLoop — SweMini-based, registered for verl
# ---------------------------------------------------------------------------


@register("django_agent_loop")
class SWEMiniDjangoAgentLoop(AgentLoopBase):
    """SweMini-style multi-turn Django agent loop with IDEGym server interaction."""

    def __init__(self, trainer_config, server_manager, tokenizer, processor, dataset_cls, data_config, **kwargs):
        super().__init__(trainer_config, server_manager, tokenizer, processor, dataset_cls, data_config, **kwargs)

        rollout_cfg = self.rollout_config
        # Custom parameters live under rollout_cfg.custom (a dict)
        custom = getattr(rollout_cfg, "custom", None) or {}

        self.max_turns = custom.get("max_turns", 10)
        self.max_num_tests = custom.get("max_num_tests", 10)
        self.max_test_output_symb = custom.get("max_test_output_symb", 10_000)
        self.enable_thinking = custom.get("enable_thinking", False)

        keep_reasoning = custom.get("keep_reasoning", "none")
        if keep_reasoning not in KEEP_REASONING_OPTIONS:
            keep_reasoning = "none"
        self.keep_reasoning = keep_reasoning

        # Parsing mode
        agent_parsing_mode = custom.get("agent_parsing_mode", "toolcall")
        self.agent_parsing_mode = agent_parsing_mode

        # Prompts
        prompts_file = custom.get("prompts_file", None)
        self.agent_prompts = load_prompts(agent_parsing_mode, prompts_file)

        self._jinja_env = JinjaEnvironment(undefined=StrictUndefined)
        self._jinja_env.filters["tojson"] = lambda v, *a, **kw: json.dumps(v, ensure_ascii=False)

        format_error_tpl = self.agent_prompts["format_error_template"]
        if agent_parsing_mode == "toolcall":
            self._strategy: ParsingStrategy = ToolcallStrategy(format_error_tpl)
        elif agent_parsing_mode == "text":
            self._strategy = TextStrategy(format_error_tpl)
        else:
            raise KeyError(f"Unknown agent_parsing_mode: {agent_parsing_mode}. Use: toolcall or text")

        # IDEGym runner
        use_mock_runner = custom.get("use_mock_runner", False)
        if use_mock_runner:
            from examples.django_idegym.agent_loop.mock_idegym_runner import MockIDEGymRunner
            mock_error_rate = custom.get("mock_runner_error_rate", 0.001)
            mock_pass_rate = custom.get("mock_runner_pass_rate", 0.7)
            logger.info(f"[SCAFFOLD] Using MockIDEGymRunner (error_rate={mock_error_rate}, pass_rate={mock_pass_rate})")
            self.idegym_runner = MockIDEGymRunner(error_rate=mock_error_rate, pass_rate=mock_pass_rate)
        else:
            from examples.django_idegym.agent_loop.idegym_runner import IDEGymRunner
            self.idegym_runner = IDEGymRunner()

        # S3 trajectory saving
        self.fs = None
        self.traj_bucket = None
        trajectory_dir = custom.get("trajectory_dir", None)
        if trajectory_dir:
            try:
                import s3fs
                self.fs = s3fs.S3FileSystem()
                trajectory_dir = trajectory_dir.rstrip("/")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                experiment_name = getattr(self.config.trainer, "experiment_name", "default")
                self.traj_bucket = f"{trajectory_dir}/traj_django_swemini_{experiment_name}_{timestamp}.jsonl"
                print(f"[INFO] Trajectories will be saved to {self.traj_bucket}")
            except Exception as e:
                print(f"[WARNING] Failed to initialize S3 for trajectory saving: {e}")

    # --- Prompt rendering ---

    def _jinja_render(self, template_str: str, **kwargs) -> str:
        return self._jinja_env.from_string(template_str).render(**kwargs)

    def _get_hint(self, item: dict) -> str:
        hint_level = item.get("hint_level", 0)
        if hint_level == 0:
            return ""
        method_body = item["method"]["body"]
        method_declaration = item["method"]["declaration"]
        if hint_level == 4:
            return method_declaration + "\n" + method_body
        lines = method_body.splitlines()
        total_lines = len(lines)
        ratio = hint_level / 4.0
        num_lines = max(1, round(total_lines * ratio))
        return method_declaration + "\n" + "\n".join(lines[:num_lines])

    def _render_instance_prompt(self, item: dict) -> str:
        method = item["method"]
        task = self._jinja_render(
            self.agent_prompts["task_template"],
            class_name=item["class_name"],
            file_path=item["file_path"],
            method_declaration=method["declaration"],
            method_description=method["description"],
        )
        hint_level = item.get("hint_level", 0)
        hint = self._get_hint(item)
        if hint and hint_level == 4:
            task += "\n\n" + (f"You're given a full solution:\n```python\n{hint}```\n"
                              "This is full solution, use it to produce the final answer.")
        elif hint:
            task += "\n\n" + (f"You're given a part of the method:\n```python\n{hint}```\n"
                              "This is part of the solution, use it to produce the final answer.")
        return self._jinja_render(
            self.agent_prompts["instance_template"],
            task=task,
            **_CONTAINER_SYSTEM_INFO,
        )

    def _render_observation(self, result: dict[str, Any]) -> str:
        cmd_output = result.get("command_output", {})
        stdout = cmd_output.get("stdout", "") or ""
        stderr = cmd_output.get("stderr", "") or ""
        output = f"{stdout}\n{stderr}".strip()
        out_obj = {
            "returncode": int(cmd_output.get("exit_code", 0)),
            "output": output,
            "exception_info": None,
        }
        return self._jinja_render(
            self.agent_prompts["observation_template"],
            output=out_obj,
            max_output_length=self.max_test_output_symb,
            head_tail_length=self.max_test_output_symb // 2,
        )

    async def _generate_messages(self, dp_item: dict) -> list[BaseMessage]:
        system_prompt = self._jinja_render(self.agent_prompts["system_template"])
        user_prompt = self._render_instance_prompt(dp_item)
        return [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]

    # --- ItemToRun builder ---

    def build_item(self, item: dict, code: str = "") -> ItemToRun:
        method = item["method"]
        if "_num_decorators" not in method:
            def_line_idx = method["global_method_declaration_index"][0]
            file_lines = item["raw_file_content"].splitlines()
            method["_num_decorators"] = count_decorators_above(file_lines, def_line_idx)
        startline = method["global_method_declaration_index"][0] + 1 - method["_num_decorators"]
        endline = method["global_method_body_index"][1] + 1
        tests = item["tests"]["full_paths"]
        if tests and self.max_num_tests > 0:
            tests = tests[:self.max_num_tests]
        return ItemToRun(
            idx=item["idx"],
            dp_id=item["dp_id"],
            file_path=item["file_path"],
            replace_content=code,
            method_name=method["name"],
            start_line=startline,
            end_line=endline,
            tests=tests,
        )

    # --- Graph nodes ---

    async def _initialize(self, state: SweMiniState, config: RunnableConfig) -> SweMiniState:
        """Node: Create IDEGym server and cut the target method."""
        agent_config = config["configurable"]["agent_config"]
        init_start_time = time.perf_counter()

        # Initialize state defaults
        state["responses_raw"] = list(state.get("responses_raw", []))
        state["test_results"] = []
        state["test_status"] = None
        state["generated_code_blocks"] = []
        state["turn"] = 0
        state["max_turns"] = agent_config["max_turns"]
        state["should_continue"] = None
        state["stop_reason"] = None
        state["edited_file"] = []
        state["no_code_block"] = None
        state["test_passed"] = None
        state["test_output"] = None
        state["tests_passed_percentage"] = None
        state["turn_exit"] = None
        state["timestamp_start"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        state["timestamp_end"] = None
        state["duration"] = None
        state["turn_tests_durations"] = []
        state["turn_gen_durations"] = []
        state["reward_components"] = None
        state["rm_score"] = FAILED_PIPELINE_SCORE
        state["client"] = None
        state["server"] = None
        state["server_id"] = None
        state["server_init_failed"] = None
        state["is_failed_rollout"] = False
        state["trajectory"] = []
        state["metadata"] = None
        state["submit_mode"] = None
        state["tools_kwargs"] = {}
        state["extra_info"] = {}

        item = state["dp_item"]
        if item is None:
            state["should_continue"] = False
            state["is_failed_rollout"] = True
            state["stop_reason"] = "no_dp_item"
            state["messages"] = [get_fallback_message("No datapoint provided.", role="ai")]
            return state

        dp_id = item["dp_id"]

        # Generate initial messages
        state["messages"] = await self._generate_messages(item)

        # Create client
        try:
            client = await self.idegym_runner.create_client()
            state["client"] = client
        except Exception as e:
            logger.error(f"[INIT] dp_id={dp_id} - Failed to create client: {e}")
            state["should_continue"] = False
            state["is_failed_rollout"] = True
            state["stop_reason"] = "initialization_error"
            state["server_init_failed"] = 1.0
            state["messages"].append(get_fallback_message("Failed initialize test client.", role="ai"))
            return state

        # Create server
        try:
            server = await self.idegym_runner.create_server(client)
            state["server"] = server
            state["server_id"] = server.server_id
        except Exception as e:
            logger.error(f"[INIT] dp_id={dp_id} - Failed to create server: {e}")
            try:
                await self.idegym_runner.close_client(client)
            except Exception:
                pass
            state["client"] = None
            state["should_continue"] = False
            state["is_failed_rollout"] = True
            state["stop_reason"] = "initialization_error"
            state["server_init_failed"] = 1.0
            state["messages"].append(get_fallback_message("Failed initialize test server.", role="ai"))
            return state

        # Cut the target method
        item_to_cut = self.build_item(item, code="")
        try:
            result = await self.idegym_runner.edit_file(server, item_to_cut)
            self._record(state, "edit_file", result, item=item_to_cut.to_dict(), dp_id=dp_id)
        except Exception as e:
            logger.error(f"[INIT] dp_id={dp_id} - Failed to cut method: {e}")
            await self.idegym_runner.finish_server(server)
            await self.idegym_runner.close_client(client)
            state["client"] = None
            state["server"] = None
            state["should_continue"] = False
            state["is_failed_rollout"] = True
            state["stop_reason"] = "initialization_error"
            state["server_init_failed"] = 1.0
            state["messages"].append(get_fallback_message("Failed initialize test server.", role="ai"))
            return state

        state["should_continue"] = True
        logger.info(f"[INIT] dp_id={dp_id} - Initialize completed in {time.perf_counter() - init_start_time:.2f}s")
        return state

    async def _agent_step(self, state: SweMiniState, config: RunnableConfig, model) -> SweMiniState:
        """Node: One iteration of the agent loop."""
        agent_config = config["configurable"]["agent_config"]
        sampling_params = config["configurable"]["sampling_params"]

        item = state["dp_item"]
        dp_id = item["dp_id"]
        state["turn"] += 1
        step = state["turn"]

        if step > self.max_turns:
            state["stop_reason"] = "max_turns_reached"
            state["should_continue"] = False
            return state

        # Call model
        state = await call_model(
            state, model=model, sampling_params=sampling_params,
            keep_reasoning=agent_config["keep_reasoning"],
            max_turns=agent_config["max_turns"],
            enable_trim=False,
        )

        if not state["should_continue"]:
            return state

        messages = state["messages"]
        if not messages:
            state["stop_reason"] = "empty_messages"
            state["should_continue"] = False
            state["is_failed_rollout"] = True
            messages.append(get_fallback_message("No messages in the state.", role="ai"))
            return state

        message: AIMessage = messages[-1]

        # Parse actions
        try:
            actions = self._strategy.parse_actions(message)
        except FormatError as e:
            if not e.messages:
                fallback = get_fallback_message("Format error occurred, please try again.", role="user")
            else:
                fallback = get_fallback_message(e.messages[0]["content"], role="user")
            if step < self.max_turns:
                messages.append(fallback)
            state["should_continue"] = True
            state["messages"] = messages
            return state
        except Exception:
            traceback.print_exc()
            raise

        # Execute commands
        commands = [a["command"] for a in actions]
        observations, stop = await self._execute_commands(commands, state)

        if stop:
            state["should_continue"] = False
            state["messages"] = messages
            return state

        # Build observation messages
        obs_msgs = [
            self._strategy.convert_msg_to_lc(msg)
            for msg in self._strategy.build_observation_msgs(actions, observations)
        ]
        messages.extend(obs_msgs)
        state["messages"] = messages
        state["should_continue"] = True
        return state

    async def _execute_commands(self, commands: list[str], state: SweMiniState) -> tuple[list[str], bool]:
        """Execute bash commands sequentially via IDEGym."""
        server = state["server"]
        step = state["turn"]
        observations: list[str] = []
        dp_id = state["dp_item"]["dp_id"]

        for cmd_idx, command in enumerate(commands):
            has_submit = SUBMIT_SIGNAL in command

            if has_submit and command.strip() == SUBMIT_SIGNAL:
                state["stop_reason"] = "submitted"
                state["submit_mode"] = "standalone"
                return observations, True

            try:
                result = await self.idegym_runner.run_bash(server, command)
                self._record(state, "run_bash", result, command=command, step=step, cmd_idx=cmd_idx, dp_id=dp_id)
                observations.append(self._render_observation(result))
            except Exception as e:
                exc_str = str(e).lower()
                if "idegym" in exc_str or "timeout" in exc_str:
                    state["stop_reason"] = "run_bash_error"
                    state["test_status"] = "CRUSHED"
                    state["should_continue"] = False
                    state["is_failed_rollout"] = True
                    state["messages"].append(get_fallback_message("IDEGYM error during bash command.", role="ai"))
                    return observations, True
                error_msg = f"Command execution failed: {type(e).__name__}: {e}"
                tpl = self.agent_prompts["format_error_template"]
                observations.append(self._jinja_render(tpl, error=error_msg, actions=[]))
                continue

            if has_submit:
                state["stop_reason"] = "submitted"
                state["submit_mode"] = "combined"
                return observations, True

        return observations, False

    async def _run_tests(self, state: SweMiniState) -> SweMiniState:
        """Node: Run tests on the server after agent submission."""
        item = state["dp_item"]
        dp_id = item["dp_id"]
        server = state["server"]

        if state["test_status"] == "CRUSHED":
            return state

        item_to_test = self.build_item(item, code="")
        try:
            test_start = time.perf_counter()
            raw_result = await self.idegym_runner.run_tests(server=server, item=item_to_test, do_edit=False)
            test_duration = time.perf_counter() - test_start
            logger.info(f"[TESTS] dp_id={dp_id} - Tests completed in {test_duration:.2f}s")

            self._record(state, "run_item", raw_result, item=item_to_test.to_dict(), dp_id=dp_id)

            test_result = self._parse_test_result(raw_result)
            state["test_results"] = [test_result]
            state["test_status"] = test_result["status"]
        except Exception as e:
            logger.error(f"[TESTS] dp_id={dp_id} - Error running tests: {e}")
            traceback.print_exc()
            state["test_status"] = "CRUSHED"

        return state

    def _parse_test_result(self, raw_result: dict[str, Any]) -> dict[str, Any]:
        test_output = raw_result.get("test_output", "")
        raw_result.pop("datapoint", None)
        parsed = parse_idegym_tests_output(test_output)
        raw_result["summary"] = parsed.pop("summary")
        raw_result["details"] = parsed.get("tests", [])
        if raw_result["summary"] is not None:
            raw_result["status"] = raw_result["summary"]["status"]
        else:
            raw_result["status"] = "FAILED"
        raw_result["percentage_passed"] = get_percentage_passed(raw_result)
        raw_result["test_output"] = test_output
        return raw_result

    def _calculate_reward_score(self, state: SweMiniState) -> dict:
        test_results = state.get("test_results", [])
        test_result = test_results[-1] if test_results else {}
        percentage_passed = test_result.get("percentage_passed", 0.0)
        return {
            "percentage_passed": percentage_passed,
            "efficiency": 0.0,
            "score": percentage_passed,
        }

    async def _finalize(self, state: SweMiniState) -> SweMiniState:
        """Node: Stop server, compute reward, save trajectory."""
        server = state.get("server")
        client = state.get("client")
        dp_id = state["dp_item"]["dp_id"] if state.get("dp_item") else "unknown"

        # Strip trailing non-AI messages beyond max_turns
        step = state["turn"]
        if step > self.max_turns:
            messages = state["messages"]
            while messages and messages[-1].type != "ai":
                messages.pop()
            state["messages"] = messages

        if server is None and client is None:
            return state

        state["timestamp_end"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if state.get("timestamp_start"):
            start = datetime.strptime(state["timestamp_start"], "%Y-%m-%d %H:%M:%S")
            end = datetime.strptime(state["timestamp_end"], "%Y-%m-%d %H:%M:%S")
            state["duration"] = (end - start).total_seconds()

        # Calculate reward
        reward_components = self._calculate_reward_score(state)
        state["rm_score"] = reward_components["score"]
        state["reward_components"] = reward_components
        state["tests_passed_percentage"] = reward_components["percentage_passed"]

        # Stop server
        if server:
            await self.idegym_runner.finish_server(server)
            state["server"] = None

        # Close client
        if client:
            try:
                await self.idegym_runner.close_client(client)
            except Exception as e:
                logger.error(f"[FINALIZE] dp_id={dp_id} - Error closing client: {e}")
            state["client"] = None

        # Save metadata
        dp_item = state["dp_item"]
        state["dp_item"] = None
        state["metadata"] = {
            "difficulty": dp_item.get("difficulty") if dp_item else None,
            "dp_id": dp_item.get("dp_id") if dp_item else None,
            "dp_idx": dp_item.get("idx") if dp_item else None,
        }

        # Save trajectory to S3
        if self.traj_bucket and self.fs:
            try:
                state_dict = dataclass_to_dict(copy.deepcopy(state))
                for mes in state_dict.get("messages", []):
                    if isinstance(mes, dict):
                        mes.pop("response_metadata", None)
                with self.fs.open(self.traj_bucket, "a", encoding="utf-8") as f:
                    json.dump(state_dict, f)
                    f.write("\n")
            except Exception as e:
                logger.error(f"[FINALIZE] dp_id={dp_id} - Failed to upload trajectory: {e}")

        return state

    # --- Trajectory recording ---

    def _record(self, state: SweMiniState, action: str, result: dict | None, **extra: Any) -> None:
        server_id = -1
        result_keys = []
        success = False
        if result is not None and isinstance(result, dict):
            server_id = result.get("server_id")
            result_keys = list(result.keys())
            success = True
        state["trajectory"].append({
            "action": action,
            "timestamp": time.time(),
            "server_id": server_id,
            "success": success,
            **extra,
            "result_keys": result_keys,
        })

    # --- Graph construction ---

    def _build_graph(self, model) -> Any:
        workflow = StateGraph(SweMiniState)

        agent_step_with_model = partial(self._agent_step, model=model)

        workflow.add_node("initialize", self._initialize)
        workflow.add_node("agent_step", agent_step_with_model)
        workflow.add_node("run_tests", self._run_tests)
        workflow.add_node("finalize", self._finalize)

        workflow.add_conditional_edges(
            "initialize",
            lambda s: s["should_continue"],
            {True: "agent_step", False: "finalize"},
        )
        workflow.add_conditional_edges(
            "agent_step",
            lambda s: s["should_continue"],
            {True: "agent_step", False: "run_tests"},
        )
        workflow.add_edge("run_tests", "finalize")
        workflow.add_edge("finalize", END)

        workflow.set_entry_point("initialize")
        return workflow.compile()

    # --- verl AgentLoopBase interface ---

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        dp_item = kwargs.get("dp_item") or kwargs.get("tools_kwargs", {}).get("item")

        model_path = self.config.actor_rollout_ref.model.path
        model_name = "/".join(model_path.split("/")[-2:])

        rollout = self.rollout_config
        model = CustomChatModel(
            model=model_name,
            client=self.server_manager,
            tokenizer=self.tokenizer,
            max_tokens=rollout.response_length,
            max_parallel_calls=rollout.multi_turn.max_parallel_calls,
            tool_parser=rollout.multi_turn.format,
            enable_thinking=self.enable_thinking,
        )

        # Bind tools based on parsing strategy
        model = self._strategy.bind_engine(model)

        config = {
            "configurable": {
                "sampling_params": sampling_params,
                "agent_config": {
                    "max_turns": self.max_turns,
                    "max_num_tests": self.max_num_tests,
                    "max_test_output_symb": self.max_test_output_symb,
                    "keep_reasoning": self.keep_reasoning,
                },
            }
        }

        graph = self._build_graph(model)

        graph_input: dict[str, Any] = {"messages": messages}
        if dp_item is not None:
            graph_input["dp_item"] = dp_item

        state = await graph.ainvoke(input=graph_input, config=config)

        # Convert to AgentLoopOutput
        output = convert_to_agent_output(state["messages"], rollout.response_length)

        # Propagate reward
        output.reward_score = float(state["rm_score"]) if state["rm_score"] is not None else None
        default_reward_info = {"percentage_passed": 0.0, "efficiency": 0.0, "score": 0.0}
        if state["reward_components"]:
            reward_info = dict(state["reward_components"])
            reward_info.setdefault("percentage_passed", 0.0)
            reward_info.setdefault("score", 0.0)
        else:
            reward_info = default_reward_info
        output.extra_fields["reward_extra_info"] = reward_info

        return output
