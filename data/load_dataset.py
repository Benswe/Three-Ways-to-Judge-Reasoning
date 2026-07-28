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
IGNORE_RATING = -100

# for huggingface dataset library parsing
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
    """Create a deterministic short id from text."""
    # wont change between interpreter sessions
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
    if isinstance(chosen_index, int) and 0 <= chosen_index < len(completions):
        rating = completions[chosen_index].get("rating")
        # -100 if labeled as None, otherwise actually label
        return IGNORE_RATING if rating is None else int(rating)
    
    # At the first detected error, chosen_completion can be null
    normalized_generated_step = generated_step.strip()
    # we must search for the completion to get rating for generated step
    for completion in completions:
        # extract and normalize the text
        completion_text = str(completion.get("text", "")).strip()
        if completion_text == normalized_generated_step:
            rating = completion.get("rating")
            return IGNORE_RATING if rating is None else int(rating)

    # Single completion fallback case
    if len(completions) == 1:
        rating = completions[0].get("rating")
        return IGNORE_RATING if rating is None else int(rating)
    
    # human-completion fallback

    human_completion = annotation.get("human_completion")
    if human_completion is not None:
        if isinstance(human_completion, dict):
            rating = human_completion.get("rating")
            # human rating is assumed to be presumed valid(1)
            return 1 if rating is None else int(rating)
    

    return IGNORE_RATING

# decide whether one raw PRM800K row contains a usable reasoning trace 
def is_usable_example(example: dict[str, Any]) -> bool:
    """Remove metadata examples and traces unsuitable for our comparison."""
    # We'll use this with the huggingface dataset .filter

    # Remove qs for annotator
    if example.get("is_quality_control_question", False):
        return False

    if example.get("is_initial_screening_question", False):
        return False

    question = example.get("question") or {}
    label = example.get("label") or {}

    problem = question.get("problem")
    generated_steps = question.get("pre_generated_steps")
    generated_answer = question.get("pre_generated_answer")
    ground_truth_answer = question.get("ground_truth_answer")
    finish_reason = label.get("finish_reason")

    # problem should be a string
    if not isinstance(problem, str) or not problem.strip():
        return False

    # there should be step(s) in list format
    if not isinstance(generated_steps, list) or len(generated_steps) == 0:
        return False
    
    # make sure all steps are strings 
    if not all(
        isinstance(step, str) and step.strip()
        for step in generated_steps
    ):
        return False
    

    if generated_answer is None:
        return False
    
    if ground_truth_answer is None:
        return False
    
    if finish_reason not in VALID_FINISH_REASONS:
        return False
    

    # if passes all conditions above is valid
    return True
    
def normalize_example(example: dict[str, Any]) -> dict[str, Any]:
    """Convert one nested PRM800K row into our shared project schema."""
    question = example["question"]
    label = example["label"]

    problem = question["problem"].strip()
    generated_steps = [
        # step should already be a string from earlier check in is_usable_example
        step.strip() for step in question["pre_generated_steps"]
    ]

    annotations = label.get("steps") or []

    step_ratings: list[int] = []
    
    # get label by matching annotations with generated steps
    for step_index, generated_step in enumerate(generated_steps):
        if step_index < len(annotations):
            rating = extract_step_rating(
                annotation=annotations[step_index],
                generated_step=generated_step
            )
        else:
            # Labeling stops after first detected error
            # mask with -100
            rating = IGNORE_RATING
        
        step_ratings.append(rating)
    
    trace_text = problem + "\n" + "\n".join(generated_steps)

    # normalized setup for each example
    return {
        # create deterministic hash for problem and problem + reasoning trace 
        "problem_id": stable_hash(problem),
        "trace_id": stable_hash(trace_text),
        "problem" : problem,
        "ground_truth_answer": str(question["ground_truth_answer"]),
        "generated_answer": str(question["pre_generated_answer"]),
        "steps": generated_steps,
        "step_ratings": step_ratings,
        "finish_reason": label["finish_reason"],
        "generation": example.get("generation"),
    }

def limit_dataset(
        dataset: Dataset,
        max_samples: int | None,
        seed: int = 42,

) -> Dataset:
    """Shuffled subset of dataset for inexpensive debugging"""
    if max_samples is None or max_samples >= len(dataset):
        return dataset

    return dataset.shuffle(seed=seed).select(range(max_samples))

def load_prm800k(
        max_train_samples: int | None = None,
        max_test_samples:int | None = None,
        seed: int = 42,
        # for huggingface multiprocessing capabilities
        num_proc: int | None = None,
) -> DatasetDict:
    """
    Download and normalize the offical Phase 2 PRM800K data.
    
    The returned DataSetDict has train and test splits using one shared schema.
    
    """
    
    # load
    raw_dataset = load_dataset(
        "json",
        data_files=DATA_FILES
    )

    # empty dataset dict for procesed schema 
    processed = DatasetDict()
    
    sample_limits = {
        "train": max_train_samples,
        "test": max_test_samples
    }

    
    for split_name, split_dataset in raw_dataset.items():
        # filter out unusable rows 
        split_dataset = split_dataset.filter(
            is_usable_example,
            num_proc=num_proc,
            # for the progress bar in command-line
            desc=f"Filtering {split_name}"
        )

        # next operation will normalize and needs to remove original cols
        # to pave way for normalized cols
        original_columns = split_dataset.column_names

        split_dataset = split_dataset.map(
            normalize_example,
            remove_columns=original_columns,
            num_proc=num_proc,
            desc=f"Normalizing {split_name}",
        )
        
        # apply sample limit
        split_dataset = limit_dataset(
            dataset=split_dataset,
            max_samples=sample_limits[split_name],
            seed=seed,
        )

        processed[split_name] = split_dataset
    
    return processed


def print_summary(dataset: DatasetDict) -> None:
    """Print simple checks that help catch preprocessing mistakes"""

    # loop over train and test splits
    for split_name, split_dataset in dataset.items():
        finish_reasons = split_dataset["finish_reason"]

        solved = sum(reason == "solution" for reason in finish_reasons)
        found_error = sum(reason == "found_error" for reason in finish_reasons)

        # count number of labeled and ignored steps
        labeled_steps = 0
        ignored_steps = 0

        for ratings in split_dataset["step_ratings"]:
            labeled_steps += sum(rating != IGNORE_RATING for rating in ratings)
            ignored_steps += sum(rating == IGNORE_RATING for rating in ratings)
        
        print(f"\n{split_name.upper()}")
        print(f"    Traces: {len(split_dataset):,}")
        print(f"    finish_reason=solution: {solved:,}")
        print(f"    finish_reason=found_error: {found_error:,}")
        print(f"    Labeled steps: {labeled_steps:,}")
        print(f"    Unlabeled steps: {ignored_steps:,}")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and normalize Phase 2 PRM800K."
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/prm800k_phase2"),
    )

    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=1000
    )

    parser.add_argument(
        "--max-test-samples",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--num-proc",
        type=int,
        default=None
    )

    return parser.parse_args()


def main() -> None:
    # read command line args
    args = parse_args()

    # load and preprocess PRM800K
    dataset = load_prm800k(
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples,
        seed=args.seed,
        num_proc=args.num_proc
    )

    
    print_summary(dataset)

    # create output directory
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(args.output_dir))

    # for check
    sample = dataset["train"][0]

    print("\nEXAMPLE NORMALIZED ROW")
    # convert to JSON text, allow for unicode
    print(json.dumps(sample, indent=2, ensure_ascii=False))
    print(f"\nSaved dataset to: {args.output_dir}")

if __name__ == "__main__":
    main()