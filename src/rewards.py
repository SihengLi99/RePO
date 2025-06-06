import re

from functools import partial, update_wrapper
from typing import Callable, Dict, Optional

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify


def accuracy_reward(completions: list[list[dict[str, str]]], solution: list[str], **kwargs) -> list[Optional[float]]:
    """Reward function that checks if the completion is the same as the ground truth."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(contents, solution):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
        )
        if len(gold_parsed) != 0:
            # We require the answer to be provided in correct latex (no malformed operators)
            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed="all",
                            units=True,
                        ),
                        # Ensures that boxed is tried first
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )
            # Compute binary rewards if verifiable, `None` otherwise to skip this example
            try:
                reward = float(verify(gold_parsed, answer_parsed))
            except Exception as e:
                print(f"verify failed: {e}, answer: {answer_parsed}, gold: {gold_parsed}")
                reward = None
        else:
            # If the gold solution is not parseable, we assign `None` to skip this example
            reward = None
            print("Failed to parse gold solution: ", sol)
        rewards.append(reward)

    return rewards

def format_reward(completions, **kwargs):
    """Reward function that checks the content starts with <think> and contains a closing </think> tag somewhere thereafter."""
    pattern = r"^<think>[\s\S]*?</think>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

REWARD_FUNCS_REGISTRY = {
    "format": format_reward,
    "accuracy": accuracy_reward,
}


if __name__ == "__main__":
    # Simple tests for format_reward
    test_completions = [
        [{"content": "<think>\nTest reasoning\n</think>\n<answer>\n1\n</answer>"}],
        [{"content": "Invalid output without tags"}],
    ]
    fmt_rewards = format_reward(test_completions)
    print("Format Reward Tests:", fmt_rewards)

    # Simple tests for accuracy_reward
    test_completions_acc = [
        [{"content": "\\boxed{1+1=2}"}],
        [{"content": "\\boxed{2+2=5}"}],
    ]
    solutions = ["\\boxed{1+1=2}", "\\boxed{2+2=4}"]
    acc_rewards = accuracy_reward(test_completions_acc, solutions)
    print("Accuracy Reward Tests:", acc_rewards)
