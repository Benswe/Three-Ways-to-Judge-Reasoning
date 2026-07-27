# 1. Download the raw reasoning traces 
# 2. Convert the deeply nested PRM800K format into a simple shared format
# 3. Save enough information for ORM, preference RM and PRM builders
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset




# -1, 0, 1 are real PRM800K ratings
# reserve -100 for steps that were never labeled
IGNORE_SCORE = -100

DATA_FILES = {
    "train" :(
        "https://github.com/openai/prm800k/raw/refs/heads/main/prm800k/data/phase2_train.jsonl"
    ),
    "test": (
        "https://github.com/openai/prm800k/raw/refs/heads/main/prm800k/data/phase2_test.jsonl"
    )
}

VALID_FINISH_REASONS = {"solution", "found_error"}

def stable_hash(text: str, length: int = 16) -> str:
    """Create a deterministic short identifier from text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]

def extract_step_rating(
        annotation: dict[str, Any],
        generated_step: str
) -> int:
    """
    Find the human rating corresponding to one generated step.

    Ratings:
        +1: correct and makes progress
        0: not incorrect, but does not make useful progress
        -1: incorrect
        -100: unlabeled
    
    
    """
    completions = annotation.get("completions") or []
    chosen_index = annotation.get("chosen_completion")


    # Normal case: the annotation identifies the selected completion
    