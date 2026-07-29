from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import DatasetDict, load_from_disk

from grading.grader import grade_answer


def get_outcome_label(
        generated_answer: str,
        ground_truth_answer: str,

) -> int:
    """
    Return 1 if the final answer is correct, otherwise 0
    """
    is_correct = grade_answer(
        given_answer=generated_answer,
        ground_truth=ground_truth_answer
    )


    return int(is_correct)

def format_orm_text(
        problem: str,
        steps: list[str],
        generated_answer: str
):
    """Format one complete reasonign trace for the ORM"""
    formatted_steps = "\n".join(
        f"Step {step_index}: {step.strip()}"
        for step_index, step in enumerate(steps, start=1)
    )
    # orm needs a string because a transformer receives a sequence of tokens
    return (
        f"Problem:\n{problem.strip()}\n\n"
        f"Solution:\n{formatted_steps}\n\n"
        f"Final answer:\n{generated_answer.strip()}"
    )


def build_orm_example(
        example: dict[str, Any],
) -> dict[str, Any]:
    """Convert one normalized reasoning trace into an ORM example."""

    text = format_orm_text(
        problem=example["problem"],
        steps=example["steps"],
        generated_answer=example["generated_answer"],
    )

    label = get_outcome_label(
        generated_answer=example["generated_answer"],
        ground_truth_answer=example["ground_truth_answer"],
    )


    return {
        "problem_id": example["problem_id"],
        "trace_id": example["trace_id"],
        "text": text,
        "label": label,
        "generated_answer": example["generated_answer"],
        "ground_truth_answer": example["ground_truth_answer"],
        "finish_reason": example["finish_reason"],
        "generation": example.get("generation"),
    }

def build_orm_dataset(
        dataset: DatasetDict,
        num_proc: int | None = None,
) -> DatasetDict:
    """Convert a normalized reasoning dataset into an ORM dataset."""
    

