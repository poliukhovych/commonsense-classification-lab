"""Download a pinned SNLI revision and audit sentence pairs."""

import hashlib
import json
import unicodedata
from itertools import combinations
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/processed/snli"
REPORTS = ROOT / "reports/metrics/snli"
CACHE = ROOT / "data/raw/hf-cache"

DATASET_ID = "stanfordnlp/snli"
LABELS = {0: "entailment", 1: "neutral", 2: "contradiction"}


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def pair_key(premise: str, hypothesis: str) -> str:
    """Hash an ordered pair; reversing its direction changes the key."""
    payload = json.dumps(
        [premise.casefold(), hypothesis.casefold()],
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)

    source_file = REPORTS / "source.json"
    if source_file.exists():
        source = json.loads(source_file.read_text(encoding="utf-8"))
        if source["dataset_id"] != DATASET_ID:
            raise ValueError("Unexpected dataset in source.json")
        revision = source["revision"]
    else:
        revision = HfApi().dataset_info(DATASET_ID).sha
        if not revision:
            raise ValueError("Could not resolve dataset revision")
        source = {"dataset_id": DATASET_ID, "revision": revision}
        source_file.write_text(
            json.dumps(source, indent=2) + "\n",
            encoding="utf-8",
        )

    dataset = load_dataset(
        DATASET_ID,
        "plain_text",
        revision=revision,
        cache_dir=str(CACHE),
    )

    if dataset["train"].features["label"].names != list(LABELS.values()):
        raise ValueError("Unexpected label mapping")

    frames = {}
    summary = []

    for split in ("train", "validation", "test"):
        frame = dataset[split].to_pandas()

        if not {"premise", "hypothesis", "label"} <= set(frame.columns):
            raise ValueError(f"{split}: unexpected columns")

        frame.insert(
            0,
            "example_id",
            [f"{split}:{index}" for index in range(len(frame))],
        )
        raw_rows = len(frame)
        invalid_label = ~frame["label"].isin(LABELS)
        missing_text = frame[["premise", "hypothesis"]].isna().any(axis=1)

        valid = frame.loc[~invalid_label & ~missing_text].copy()
        valid["premise"] = valid["premise"].map(normalize)
        valid["hypothesis"] = valid["hypothesis"].map(normalize)

        empty_text = valid["premise"].eq("") | valid["hypothesis"].eq("")
        empty_count = int(empty_text.sum())
        valid = valid.loc[~empty_text].copy()

        valid["label"] = valid["label"].astype(int)
        valid["label_name"] = valid["label"].map(LABELS)
        valid["pair_key"] = [
            pair_key(premise, hypothesis)
            for premise, hypothesis in zip(valid["premise"], valid["hypothesis"], strict=True)
        ]

        label_counts = valid.groupby("pair_key")["label"].nunique()
        premise_words = valid["premise"].str.split().str.len()
        hypothesis_words = valid["hypothesis"].str.split().str.len()

        summary.append(
            {
                "split": split,
                "raw_pairs": raw_rows,
                "invalid_labels": int(invalid_label.sum()),
                "missing_text_rows": int(missing_text.sum()),
                "empty_text_rows_after_filter": empty_count,
                "valid_pairs": len(valid),
                "entailment": int(valid["label"].eq(0).sum()),
                "neutral": int(valid["label"].eq(1).sum()),
                "contradiction": int(valid["label"].eq(2).sum()),
                "duplicate_extra_rows": int(valid["pair_key"].duplicated().sum()),
                "conflicting_pair_groups": int(label_counts.gt(1).sum()),
                "mean_premise_words": round(float(premise_words.mean()), 2),
                "mean_hypothesis_words": round(float(hypothesis_words.mean()), 2),
            }
        )

        valid.to_parquet(OUTPUT / f"{split}.parquet", index=False)
        frames[split] = valid

    overlaps = []
    for left, right in combinations(frames, 2):
        shared = set(frames[left]["pair_key"]) & set(frames[right]["pair_key"])
        overlaps.append(
            {
                "split_a": left,
                "split_b": right,
                "shared_pair_keys": len(shared),
            }
        )

    full = pd.concat(frames.values(), ignore_index=True)
    duplicate_mask = full["pair_key"].duplicated(keep=False)
    full.loc[duplicate_mask].sort_values(["pair_key", "example_id"]).to_parquet(
        OUTPUT / "duplicate_pairs.parquet", index=False
    )

    global_counts = full.groupby("pair_key")["label"].nunique()
    conflict_keys = global_counts[global_counts.gt(1)].index
    full.loc[full["pair_key"].isin(conflict_keys)].sort_values(
        ["pair_key", "example_id"]
    ).to_parquet(OUTPUT / "conflicting_pairs.parquet", index=False)

    summary_frame = pd.DataFrame(summary)
    overlap_frame = pd.DataFrame(overlaps)
    summary_frame.to_csv(REPORTS / "dataset_summary.csv", index=False)
    overlap_frame.to_csv(REPORTS / "split_overlap.csv", index=False)

    manifest = {
        **source,
        "labels": LABELS,
        "normalization": "NFC and whitespace; case and punctuation preserved",
        "duplicate_matching": "Ordered pair of casefolded normalized texts",
        "filtering": "Keep labels 0/1/2 and nonempty, nonmissing texts",
        "duplicates_removed": False,
        "training_subset_created": False,
        "global_conflicting_pair_groups": len(conflict_keys),
    }
    (REPORTS / "preparation.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nSNLI SUMMARY")
    print(summary_frame.to_string(index=False))
    print("\nCROSS-SPLIT PAIR OVERLAP")
    print(overlap_frame.to_string(index=False))
    print(f"\nGlobal conflicting pair groups: {len(conflict_keys)}")
    print(f"Dataset revision: {revision}")
    print("\nSaved to data/processed/snli/")
    print("No duplicates removed; no training subset sampled yet.")


if __name__ == "__main__":
    main()
