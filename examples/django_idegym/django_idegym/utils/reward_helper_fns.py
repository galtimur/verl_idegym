import re
from typing import Optional

from langchain_core.messages import BaseMessage

# Each output is either a float score or a dict containing a score key and some extra data
RewardOutput = float | dict


def extract_code_block(text: str, language: str = "python") -> Optional[str]:
    """
    Extracts a code block from a string delimited by triple backticks
    with the given language identifier.

    :param text: The input string that may contain a code block.
    :param language: The language identifier (default: "python").
    :return: The code block content, or None if not found.
    """
    match = re.search(rf"```{language}\s+(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else None


def apply_reasoning_filter(message: BaseMessage, max_turns: int) -> BaseMessage:
    """Remove a leading <think>...</think> reasoning trace."""

    # Try removing an opening <think> block all the way through the first closing </think>.
    filtered, num_substitutions = re.subn(r"^.*?</think>", "", message.content, flags=re.DOTALL)
    if num_substitutions > 0:
        # If we actually removed something, strip empty lines the block left behind.
        message.content = filtered.strip("\n")
        return message

    stripped = message.content.strip()
    if stripped.startswith("<think>"):
        # When there's only an opening <think>, keep original text for single-turn runs,
        # otherwise fall back to the short "be concise" message.
        message.content = (
            message.content if max_turns == 1 else "The thinking was too long. I should be more concise next time."
        )
        return message

    # No reasoning trace detected; return the content unchanged.
    return message
