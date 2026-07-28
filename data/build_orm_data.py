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

