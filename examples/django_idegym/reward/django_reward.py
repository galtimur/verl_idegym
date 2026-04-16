import asyncio
import os

import aiohttp

from examples.django_idegym.reward.idegym_runner_utils import ItemToRun
from examples.django_idegym.utils.postprocessing import (
    code_blocks_distance,
    extract_and_clean_code_block,
    is_good_start,
    normalize_indent,
    process_test_result_for_reward_computation,
    strip_decorators,
)

IDEGYM_SERVER__URL = os.environ.get(
    "IDEGYM_SERVER_URL", "http://idegym-django-api.ml4se.svc.cluster.local:7778/run_item"
)


class DjangoRewardComponents:
    """Reward components with their default values.

    Uses a simple class instead of Enum to avoid Python Enum aliasing
    (members with duplicate float values get merged).
    Each component is a (key_string, default_value) tuple.
    """

    PASSED_TEST_SCORE = ("passed_test_score", 1.0)
    FAILED_TEST_SCORE = ("failed_test_score", -1.0)
    SIMILARITY_SCORE = ("similarity_score", 0.0)
    NO_CODE_SCORE = ("no_code_score", -1.0)
    NO_METHOD_DEC_SCORE = ("no_method_dec_score", -0.5)
    PIPELINE_ERROR_SCORE = ("pipeline_error_score", -100.0)
    TEST_SCORE = ("test_score", 0.0)  # determined by the test result

    _ALL = [
        PASSED_TEST_SCORE, FAILED_TEST_SCORE, SIMILARITY_SCORE,
        NO_CODE_SCORE, NO_METHOD_DEC_SCORE, PIPELINE_ERROR_SCORE, TEST_SCORE,
    ]


def _key(component: tuple) -> str:
    return component[0]


def _default(component: tuple) -> float:
    return component[1]


RC = DjangoRewardComponents


def get_empty_feedback(*args, **kwargs) -> dict | None:
    return None


async def get_idegym_feedback_async(
    item: ItemToRun,
    do_edit: bool = True,
    timeout: float = 15 * 60,
    max_connections: int = 200,
    max_connections_per_host: int = 200,
) -> dict | None:
    """
    Async version of get_idegym_feedback with configurable timeout and connection limits.
    """
    connector = aiohttp.TCPConnector(
        limit=max_connections,
        limit_per_host=max_connections_per_host,
        keepalive_timeout=30,
        enable_cleanup_closed=True,
    )
    timeout_config = aiohttp.ClientTimeout(total=timeout)

    try:
        async with aiohttp.ClientSession(connector=connector, timeout=timeout_config) as session:
            async with session.post(
                IDEGYM_SERVER__URL, json=item.to_dict(), params={"do_edit": str(do_edit)}
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    print(
                        f"[ERROR] Failed to retrieve data, status code: {response.status}, "
                        f"item was:\n {item.to_dict()}"
                    )
                    return None
    except (asyncio.TimeoutError, aiohttp.ServerTimeoutError):
        print(f"[ERROR] Request timed out after {timeout} seconds for item: {item.to_dict()}")
        return None
    except Exception as e:
        print(f"[ERROR] Request failed with exception: {e} for item: {item.to_dict()}")
        return None


def get_score_dict_with_score_key(scores: dict) -> dict:
    """
    Compute the reward score that verl uses for learning by summing all the reward components,
    which are only used for logging. Then add that score to the scores dict with the key "score".
    """
    assert "score" not in scores

    init_reward_dict = {_key(c): 0.0 for c in RC._ALL}
    # Update all the reward components with the computed values
    merged_scores = init_reward_dict | scores

    assert all(isinstance(val, float) for val in merged_scores.values())

    score_sum = sum(merged_scores.values())
    merged_scores["score"] = float(score_sum)
    return merged_scores


def compute_django_reward_components_from_code_and_tests(
    code_block: str | None,
    item_dict: dict | None,
    test_result_processed: dict | None,
) -> dict:
    """
    Compute reward components for Django given a code string and an optional processed test result.
    This is the single source of truth for reward computation logic.

    Args:
        code_block: Code block string or None
        item_dict: Dictionary containing method information
        test_result_processed: Processed test result with "summary" field, or None

    Returns:
        Dictionary of reward components
    """
    if code_block is None:
        print("[DEBUG] code block is None")
        return {_key(RC.NO_CODE_SCORE): _default(RC.NO_CODE_SCORE)}

    code_block = code_block.replace("\r\n", "\n").replace("\r", "\n").strip("\n")
    if not code_block.strip():
        print("[DEBUG] code block is empty")
        return {_key(RC.NO_CODE_SCORE): _default(RC.NO_CODE_SCORE)}

    if item_dict is None:
        print("[DEBUG] item_dict is None")
        return {_key(RC.NO_CODE_SCORE): _default(RC.NO_CODE_SCORE)}

    method_code = item_dict["method"]["declaration"] + "\n" + item_dict["method"]["body"]
    method_name = item_dict["method"]["name"]

    # Check if code starts with method declaration (before normalization/stripping)
    good_start = is_good_start(code=code_block, method_name=method_name)

    # Normalize and strip decorators for similarity calculation
    code_block_norm = normalize_indent(code_block)
    code_block_for_similarity = strip_decorators(code_block_norm)
    code_distance = code_blocks_distance(code_block_for_similarity, method_code)
    reward_similarity = _default(RC.SIMILARITY_SCORE) * (1 - code_distance)

    if not good_start:
        return {
            _key(RC.SIMILARITY_SCORE): reward_similarity,
            _key(RC.NO_METHOD_DEC_SCORE): _default(RC.NO_METHOD_DEC_SCORE),
        }

    # No tests provided -> fallback to fail test base
    if not test_result_processed:
        return {
            _key(RC.SIMILARITY_SCORE): reward_similarity,
            _key(RC.TEST_SCORE): _default(RC.FAILED_TEST_SCORE),
        }

    summary = test_result_processed.get("summary")
    if summary is None:
        return {
            _key(RC.SIMILARITY_SCORE): reward_similarity,
            _key(RC.TEST_SCORE): _default(RC.FAILED_TEST_SCORE),
        }

    if summary.get("status") == "OK":
        return {
            _key(RC.SIMILARITY_SCORE): reward_similarity,
            _key(RC.TEST_SCORE): _default(RC.PASSED_TEST_SCORE),
        }

    total_tests = summary.get("total_tests", 0)
    details = summary.get("details", {})
    total_tests = max((details.get("failures", 0) + details.get("errors", 0)), total_tests)
    passed_tests = total_tests - (details.get("failures", 0) + details.get("errors", 0))
    passed_percentage = (passed_tests / total_tests) if total_tests > 0 else 0.0

    print(f"[DEBUG] passed_percentage: {passed_percentage}")

    failed_score = _default(RC.FAILED_TEST_SCORE)
    passed_score = _default(RC.PASSED_TEST_SCORE)
    test_score = failed_score + (passed_score - failed_score) * passed_percentage

    return {
        _key(RC.SIMILARITY_SCORE): reward_similarity,
        _key(RC.TEST_SCORE): test_score,
    }


# ---------------------------------------------------------------------------
# Batch reward functions for standalone reward path (non-agent-loop usage)
# ---------------------------------------------------------------------------


async def get_idegym_reward_async(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict,
    **reward_kwargs,
) -> dict:
    reward_components_dict = await get_idegym_reward_components_async(
        data_source, solution_str, ground_truth, extra_info,
    )
    score_dict = get_score_dict_with_score_key(reward_components_dict)
    return score_dict


async def get_idegym_reward_components_async(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict,
    **reward_kwargs,
) -> dict:
    """Calculate reward for Django method generation based on test results (async version)."""
    code_block = extract_and_clean_code_block(solution_str)
    if code_block is None:
        print("[DEBUG] code block is None")
        return {_key(RC.NO_CODE_SCORE): _default(RC.NO_CODE_SCORE)}

    item_dict = extra_info.get("tools_kwargs", {}).get("item", None)
    if item_dict is None:
        print("[ERROR] No item found in extra_info")
        return {_key(RC.PIPELINE_ERROR_SCORE): _default(RC.PIPELINE_ERROR_SCORE)}

    code_for_tests = normalize_indent(code_block)
    code_for_tests = strip_decorators(code_for_tests)
    item_to_run: ItemToRun = ItemToRun.from_item(
        item_dict, replace=code_for_tests, max_num_tests=extra_info.get("max_num_tests", 10)
    )
    test_result = await get_idegym_feedback_async(item_to_run)

    test_result_processed = process_test_result_for_reward_computation(test_result)
    if test_result_processed is None:
        print("[DEBUG] test result is None or could not be processed")
        return {_key(RC.PIPELINE_ERROR_SCORE): _default(RC.PIPELINE_ERROR_SCORE)}

    return compute_django_reward_components_from_code_and_tests(
        code_block,
        item_dict,
        test_result_processed,
    )


async def _get_idegym_rewards_async_impl(
    data_sources: list[str],
    solution_strs: list[str],
    ground_truths: list[str],
    extra_infos: list[dict],
    **reward_kwargs,
) -> list[dict]:
    """Internal async implementation that processes multiple items concurrently."""
    tasks = []
    for data_source, solution_str, ground_truth, extra_info in zip(
        data_sources, solution_strs, ground_truths, extra_infos, strict=False
    ):
        tasks.append(
            get_idegym_reward_async(
                data_source=data_source,
                solution_str=solution_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                **reward_kwargs,
            )
        )

    results = await asyncio.gather(*tasks, return_exceptions=True)

    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"[ERROR] Task {i} failed with exception: {result}")
            error_reward = {_key(c): 0.0 for c in RC._ALL}
            error_reward[_key(RC.PIPELINE_ERROR_SCORE)] = _default(RC.PIPELINE_ERROR_SCORE)
            result = get_score_dict_with_score_key(error_reward)
        processed_results.append(result)

    # Replace pipeline error scores with average valid reward for GRPO/REINFORCE
    rl_algo = os.getenv("RL_ALGORITHM", "reinforce").lower()
    pipeline_key = _key(RC.PIPELINE_ERROR_SCORE)
    if rl_algo.startswith(("reinforce", "grpo")):
        valid_rewards = [
            reward["score"]
            for reward in processed_results
            if reward[pipeline_key] == 0.0
        ]
        broken_pipe_reward = sum(valid_rewards) / len(valid_rewards) if valid_rewards else 0.0
    else:
        broken_pipe_reward = 0

    for reward in processed_results:
        if reward[pipeline_key] != 0.0:
            reward["score"] = broken_pipe_reward

    return processed_results


def get_idegym_rewards_async(
    data_sources: list[str],
    solution_strs: list[str],
    ground_truths: list[str],
    extra_infos: list[dict],
    **reward_kwargs,
) -> list[dict]:
    """
    Async version of get_idegym_rewards that processes multiple items concurrently.
    Provides a synchronous interface.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        future = asyncio.run_coroutine_threadsafe(
            _get_idegym_rewards_async_impl(
                data_sources, solution_strs, ground_truths, extra_infos, **reward_kwargs
            ),
            loop,
        )
        return future.result()
    else:
        return asyncio.run(
            _get_idegym_rewards_async_impl(
                data_sources, solution_strs, ground_truths, extra_infos, **reward_kwargs
            )
        )
