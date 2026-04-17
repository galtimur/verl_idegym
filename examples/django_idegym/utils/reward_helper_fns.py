import re
from typing import Optional

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


def apply_reasoning_filter(content: str, max_turns: int) -> str:
    """Remove a leading <think>...</think> reasoning trace from content string."""

    filtered, num_substitutions = re.subn(r"^.*?</think>", "", content, flags=re.DOTALL)
    if num_substitutions > 0:
        return filtered.strip("\n")

    stripped = content.strip()
    if stripped.startswith("<think>"):
        return (
            content if max_turns == 1 else "The thinking was too long. I should be more concise next time."
        )

    return content
