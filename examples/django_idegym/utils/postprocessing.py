import re
from typing import Any

from examples.django_idegym.utils.reward_helper_fns import extract_code_block


def normalize_indent(code: str) -> str:
    # assumes the first line is the function definition
    code = code.strip("\n")
    lines = code.splitlines()
    line = lines[0]
    indent_len = len(line) - len(line.lstrip())
    # Remove up to indent_len spaces from the start of every line
    normalized = [line[indent_len:] if len(line) >= indent_len else "" for line in lines]
    return "\n".join(normalized)


def strip_decorators(code: str) -> str:
    """
    Remove leading decorators that directly precede a function definition.
    Keeps the rest of the code unchanged.
    """
    if not code:
        return code

    lines = code.splitlines()
    lines = [line for line in lines if line.strip()]
    lines_without_decorator = []
    is_method_body = False
    for line in lines:
        if line.strip().startswith("@") and not is_method_body:
            continue
        else:
            is_method_body = True
        lines_without_decorator.append(line)
    return "\n".join(lines_without_decorator)


def is_good_start(code: str, method_name: str) -> bool:
    method_dec = f"def {method_name}"
    method_dec_async = f"async def {method_name}"
    # allow starting with decorators immediately preceding the function
    # e.g., @decorator(args)
    stripped = code.strip()
    good_start = stripped.startswith((method_dec, method_dec_async)) or stripped.startswith("@")
    return good_start


def parse_idegym_tests_output(output: str) -> dict[str, Any]:
    test_pattern = re.compile(
        r"^(={70})\n"
        r"([A-Z]+):\s"  # ===== header  # Status, eg: ERROR:
        r"([^\s]+)\s+\(([^)]+)\)\n"  # Test method and module path in parentheses
        r"(?:.*?)"
        r"[-]{70}\n"  # ------
        r"(.*?)(?=\n[=-]{70}|\Z)",  # Stacktrace (non-greedy, up to next ===== or end of file
        re.DOTALL | re.MULTILINE,
    )
    # Pattern for summary statistics at end of file
    summary_pattern = re.compile(
        r"Ran (\d+) tests? in [\d.]+s\s*\n\s*"  # Ran X tests in Y.ZZZs
        r"([A-Z]+)(?: \(([^)]+)\))?",  # FAILED or OK, optionally with details in parentheses
        re.MULTILINE,
    )
    results = []
    for match in test_pattern.finditer(output):
        status = match.group(2)
        test_method = match.group(3)  # Just the method name (test_i18n_app_dirs)
        # Module path (i18n.tests.WatchForTranslationChangesTests)
        module_path = match.group(4)
        full_test_path = f"{module_path}.{test_method}"  # Complete test path
        stacktrace = match.group(5)
        results.append(
            {
                "test_full_path": full_test_path,
                "status": status,
                "test_method": test_method,
                "module_path": module_path,
                "stacktrace": stacktrace.strip(),
            }
        )

    # Extract summary information
    summary_match = summary_pattern.search(output)
    if summary_match:
        total_tests = int(summary_match.group(1))
        overall_status = summary_match.group(2)  # "FAILED" or "OK"
        details_str = summary_match.group(3)
        # e.g. "failures=1, errors=14, skipped=998, expected failures=4"

        # Parse the details into a dictionary
        details = {}
        if details_str:
            for item in details_str.split(", "):
                key, value = item.split("=")
                details[key] = int(value)

        summary_info = {
            "total_tests": total_tests,
            "status": overall_status,
            "details": details,
        }
    else:
        summary_info = None

    return {"tests": results, "summary": summary_info}


def strip_comments_and_whitespace(code: str) -> str:
    """Remove comments and whitespace from code for comparison purposes."""
    # Remove single-line comments (Python)
    code = re.sub(r"#.*", "", code)
    # Remove multiline comments (Python)
    code = re.sub(r"'''(.|\n)*?'''", "", code)
    code = re.sub(r'"""(.|\n)*?"""', "", code)
    # Strip whitespace and blank lines
    code = "\n".join(line for line in code.splitlines() if line.strip())
    return code.strip()


def levenshtein(s1: str, s2: str) -> int:
    """Calculate Levenshtein distance between two strings."""
    # Optimize: ensure s1 is the longer string
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def normalized_levenshtein(a: str, b: str) -> float:
    """Calculate normalized Levenshtein distance (0.0 to 1.0)."""
    ed = levenshtein(a, b)
    norm = ed / max(len(a), len(b)) if max(len(a), len(b)) > 0 else 0.0
    return norm


def code_blocks_distance(code1: str, code2: str) -> float:
    """Calculate normalized distance between two code blocks after removing comments and whitespace."""
    clean1 = strip_comments_and_whitespace(code1)
    clean2 = strip_comments_and_whitespace(code2)
    return normalized_levenshtein(clean1, clean2)


def extract_and_clean_code_block(solution_str: str) -> str | None:
    """
    Extract and clean code block from solution string.

    Args:
        solution_str: The solution string that may contain code blocks

    Returns:
        Cleaned code block or None if no code block found
    """
    # Filtering possible think block
    solution_str = re.sub(
        r"^.*?</think>", "", str(solution_str), flags=re.DOTALL
    ).strip("\n")
    # Extract code block from solution string
    code_block = extract_code_block(solution_str, "python")
    if code_block is None:
        return None

    # Clean up the code block
    code_block = code_block.replace("\r\n", "\n")
    code_block = code_block.replace("\r", "\n")
    code_block = code_block.strip("\n")

    return code_block if code_block.strip() else None


def process_test_result_for_reward_computation(test_result: dict | None) -> dict | None:
    """
    Process test result into the format expected by
    compute_django_reward_components_from_code_and_tests.
    """
    if test_result is None:
        return None

    # If external caller passed a pre-processed test dict with summary/details, use it directly
    if "summary" in test_result:
        return test_result

    # Otherwise parse test_output into a processed structure
    test_output = test_result.get("test_output", "")
    if not test_output:
        return None

    parsed = parse_idegym_tests_output(test_output)
    return {"summary": parsed.get("summary"), "details": parsed.get("tests")}


def get_percentage_passed(test_results: dict) -> float:
    """Extract test pass percentage from a processed test result dict."""
    summary = test_results.get("summary", {})
    if summary is None:
        return 0.0
    total_tests = summary.get("total_tests", 0)
    details = summary.get("details", {})
    if (details is None) or (total_tests is None) or (total_tests == 0):
        return 0.0

    total_tests = max((details.get("failures", 0) + details.get("errors", 0)), total_tests)
    passed_tests = total_tests - (details.get("failures", 0) + details.get("errors", 0))
    return (passed_tests / total_tests) if total_tests > 0 else 0.0
