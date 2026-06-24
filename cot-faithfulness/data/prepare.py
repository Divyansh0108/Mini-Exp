import json
import sys
from pathlib import Path

from datasets import load_dataset

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import ARC_SAMPLE_PATH, NUM_SAMPLES, RANDOM_SEED

LABEL_MAP = {"1": "A", "2": "B", "3": "C", "4": "D", "A": "A", "B": "B", "C": "C", "D": "D"}


def normalize_record(record):
    labels = [LABEL_MAP.get(label) for label in record["choices"]["label"]]
    if any(label is None for label in labels):
        return None
    if set(labels) != {"A", "B", "C", "D"}:
        return None
    choices = dict(zip(labels, record["choices"]["text"]))
    answer = LABEL_MAP.get(record["answerKey"])
    if answer is None:
        return None
    return {
        "id": record["id"],
        "question": record["question"],
        "choices": {label: choices[label] for label in ("A", "B", "C", "D")},
        "ground_truth": answer,
    }


def main():
    dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    filtered = []
    for record in dataset:
        if len(record["choices"]["label"]) != 4 or len(record["choices"]["text"]) != 4:
            continue
        normalized = normalize_record(record)
        if normalized is not None:
            filtered.append(normalized)
    sampled = filtered[:]
    import random

    random.Random(RANDOM_SEED).shuffle(sampled)
    sampled = sampled[:NUM_SAMPLES]
    ARC_SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARC_SAMPLE_PATH.write_text(json.dumps(sampled, indent=2), encoding="utf-8")
    print(f"Saved {len(sampled)} questions to {ARC_SAMPLE_PATH}")


if __name__ == "__main__":
    main()
