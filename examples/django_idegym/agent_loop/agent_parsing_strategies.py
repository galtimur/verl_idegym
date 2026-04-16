"""
Parsing strategies for SweMini agent: toolcall vs text-based command extraction.

Ported from jetrl_django_idegym/scaffold/agent_parsing_strategies.py.
"""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from types import SimpleNamespace
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

# ---------------------------------------------------------------------------
# Vendored from minisweagent to avoid external dependency
# ---------------------------------------------------------------------------

from jinja2 import StrictUndefined, Template


class FormatError(Exception):
    """Raised when the LM's output is not in the expected format."""

    def __init__(self, *messages: dict):
        self.messages = messages
        super().__init__()


def parse_regex_actions(content: str, *, action_regex: str, format_error_template: str) -> list[dict]:
    """Parse actions from text content using regex. Raises FormatError if not exactly one action."""
    actions = [a.strip() for a in re.findall(action_regex, content, re.DOTALL)]
    if len(actions) != 1:
        error_msg = f"Expected exactly 1 action, found {len(actions)}."
        raise FormatError(
            {
                "role": "user",
                "content": Template(format_error_template, undefined=StrictUndefined).render(
                    actions=actions, error=error_msg
                ),
            }
        )
    return [{"command": action} for action in actions]


def parse_toolcall_actions(tool_calls: list, *, format_error_template: str) -> list[dict]:
    """Parse tool calls from the response. Raises FormatError if unknown tool or invalid args."""
    if not tool_calls:
        raise FormatError(
            {
                "role": "user",
                "content": Template(format_error_template, undefined=StrictUndefined).render(
                    error="No tool calls found in the response. Every response MUST include at least one tool call.",
                    actions=[],
                ),
            }
        )
    actions = []
    for tool_call in tool_calls:
        error_msg = ""
        args = {}
        try:
            args = json.loads(tool_call.function.arguments)
        except Exception as e:
            error_msg = f"Error parsing tool call arguments: {e}. "
        if tool_call.function.name != "bash":
            error_msg += f"Unknown tool '{tool_call.function.name}'."
        if "command" not in args:
            error_msg += "Missing 'command' argument in bash tool call."
        if error_msg:
            raise FormatError(
                {
                    "role": "user",
                    "content": Template(format_error_template, undefined=StrictUndefined).render(
                        actions=[], error=error_msg.strip()
                    ),
                }
            )
        actions.append({"command": args["command"], "tool_call_id": tool_call.id})
    return actions

logger = logging.getLogger(__name__)

MSWEA_BLOCK_RE = re.compile(r"```mswea_bash_command\s*\n(.*?)```", re.DOTALL)

BASH_TOOL = {
    "name": "bash",
    "description": "Execute a bash command",
    "parameters": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The bash command to execute",
            }
        },
        "required": ["command"],
    },
}


def get_message_content(message: AIMessage) -> str:
    """Get content of an AI message, stripping <think>...</think> blocks."""
    raw_content = str(message.content)
    filtered, num_subs = re.subn(r"^.*?</think>", "", raw_content, flags=re.DOTALL)
    content = filtered.strip("\n") if num_subs > 0 else raw_content
    return content


def extract_commands_from_message(msg: dict) -> list[str]:
    """Extract bash commands from a stored agent message dict."""
    commands: list[str] = []
    for tc in msg.get("tool_calls", []):
        if tc.get("name") == "bash":
            cmd = tc.get("args", {}).get("command", "")
            if cmd:
                commands.append(cmd)
    if commands:
        return commands
    content = msg.get("content", "")
    commands.extend(m.group(1).strip() for m in MSWEA_BLOCK_RE.finditer(content))
    return commands


def adapt_lc_tool_calls(lc_tool_calls: list[dict]) -> list[SimpleNamespace]:
    """Convert LangChain tool-call dicts to the litellm-style objects
    expected by ``parse_toolcall_actions``."""
    return [
        SimpleNamespace(
            id=tc.get("id", ""),
            function=SimpleNamespace(
                name=tc.get("name", ""),
                arguments=json.dumps(tc.get("args", {})),
            ),
        )
        for tc in lc_tool_calls
    ]


def convert_messages(strategy: "ParsingStrategy", agent_messages: list[dict]) -> list[BaseMessage]:
    return [strategy.convert_msg_to_lc(msg) for msg in agent_messages]


class ParsingStrategy(ABC):
    """Abstracts the differences between toolcall and text parsing modes."""

    @abstractmethod
    def bind_engine(self, engine):
        """Optionally bind tools to the LLM engine."""

    @abstractmethod
    def parse_actions(self, message: AIMessage) -> list[dict]:
        """Parse commands from model response. Raises FormatError on bad format."""

    @abstractmethod
    def build_assistant_msg(self, message: AIMessage) -> dict:
        """Build an internal assistant message dict from model response."""

    @abstractmethod
    def build_observation_msgs(self, actions: list[dict], observations: list[str]) -> list[dict]:
        """Build observation messages to append to conversation."""

    @abstractmethod
    def convert_msg_to_lc(self, msg: dict) -> BaseMessage:
        """Convert an internal message dict to a LangChain message object."""

    @abstractmethod
    def get_tools(self) -> list[dict]:
        """Get the tools available for the model."""


class ToolcallStrategy(ParsingStrategy):
    def __init__(self, format_error_template: str):
        self._format_error_template = format_error_template
        self._tools = [BASH_TOOL]

    def bind_engine(self, engine):
        return engine.bind_tools(self._tools)

    def parse_actions(self, message: AIMessage) -> list[dict]:
        adapted = adapt_lc_tool_calls(message.tool_calls or [])
        return parse_toolcall_actions(
            adapted,
            format_error_template=self._format_error_template,
        )

    def build_assistant_msg(self, message: AIMessage) -> dict:
        content = get_message_content(message)
        msg: dict[str, Any] = {"role": "assistant", "content": content}
        if message.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.get("id", ""),
                    "name": tc.get("name", ""),
                    "args": tc.get("args", {}),
                }
                for tc in message.tool_calls
            ]
        return msg

    def build_observation_msgs(self, actions: list[dict], observations: list[str]) -> list[dict]:
        msgs: list[dict] = []
        for action, obs in zip(actions, observations):
            msgs.append(
                {
                    "role": "tool",
                    "tool_call_id": action.get("tool_call_id", ""),
                    "content": obs,
                }
            )
        return msgs

    def convert_msg_to_lc(self, msg: dict) -> BaseMessage:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            return SystemMessage(content=content)
        if role == "assistant":
            if msg.get("tool_calls"):
                tc_list = [{**tc, "type": "tool_call"} for tc in msg["tool_calls"]]
                return AIMessage(content=content, tool_calls=tc_list)
            return AIMessage(content=content)
        if role == "tool":
            return ToolMessage(content=content, tool_call_id=msg.get("tool_call_id", ""))
        return HumanMessage(content=content)

    def get_tools(self) -> list:
        return self._tools


class TextStrategy(ParsingStrategy):
    _ACTION_REGEX = r"```mswea_bash_command\s*\n(.*?)```"

    def __init__(self, format_error_template: str):
        self._format_error_template = format_error_template
        self._tools = []

    def bind_engine(self, engine):
        return engine

    def parse_actions(self, message: AIMessage) -> list[dict]:
        content = get_message_content(message)
        return parse_regex_actions(
            content,
            action_regex=self._ACTION_REGEX,
            format_error_template=self._format_error_template,
        )

    def build_assistant_msg(self, message: AIMessage) -> dict:
        content = get_message_content(message)
        return {"role": "assistant", "content": content}

    def build_observation_msgs(self, actions: list[dict], observations: list[str]) -> list[dict]:
        combined = (
            "\n\n---\n\n".join(observations) if len(observations) > 1 else (observations[0] if observations else "")
        )
        if not combined:
            return []
        return [{"role": "user", "content": combined}]

    def convert_msg_to_lc(self, msg: dict) -> BaseMessage:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            return SystemMessage(content=content)
        if role == "assistant":
            return AIMessage(content=content)
        return HumanMessage(content=content)

    def get_tools(self) -> list:
        return self._tools
