"""Clean SNLI and freeze a reproducible stratified training subset."""

import hashlib
import json
from itertools import combinations
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/processed/snli"
OUTPUT = ROOT / "data/clean/snli"
REPORTS = ROOT / "reports/metrics/snli"

TRAIN_SIZE = 100_000
SEED = 42
LABELS = {0: "entailment", 1: "neutral", 2: "contradiction"}


def premise_group(text: str) -> str:
    """Input text is already NFC- and whitespace-normalized."""
    return hashlib.sha256(text.casefold().encode("utf-8")).hexdigest()


def stratified_sample(frame: pd.DataFrame) -> pd.DataFrame:
    """Allocate exact sample size proportionally using largest remainders."""
    counts = frame["label"].value_counts().sort_index()
    quotas = counts / len(frame) * TRAIN_SIZE
    allocation = quotas.astype(int)

    remainder = TRAIN_SIZE - int(allocation.sum())
    order = (quotas - allocation).sort_values(ascending=False, kind="stable").index
    for label in order[:remainder]:
        allocation.loc[label] += 1

    parts = [
        frame.loc[frame["label"].eq(label)].sample(
            n=int(allocation.loc[label]),
            random_state=SEED + int(label),
        )
        for label in counts.index
    ]
    return (
        pd.concat(parts, ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    original = {
        split: pd.read_parquet(SOURCE / f"{split}.parquet")
        for split in ("train", "validation", "test")
    }

    for split, frame in original.items():
        if not frame["example_id"].is_unique:
            raise ValueError(f"{split}: duplicate example IDs")
        if not frame["label"].isin(LABELS).all():
            raise ValueError(f"{split}: unexpected labels")

    train = original["train"]
    counts = train.groupby("pair_key")["label"].nunique()
    conflict_keys = set(counts[counts.gt(1)].index)

    heldout_keys = set(original["validation"]["pair_key"]) | set(original["test"]["pair_key"])

    removed = []

    def remove(
        frame: pd.DataFrame,
        mask: pd.Series,
        split: str,
        reason: str,
    ) -> pd.DataFrame:
        selected = frame.loc[mask, ["example_id", "pair_key", "label"]].copy()
        selected["split"] = split
        selected["reason"] = reason
        removed.append(selected)
        return frame.loc[~mask].copy()

    train = remove(
        train,
        train["pair_key"].isin(conflict_keys),
        "train",
        "conflicting_train_labels",
    )
    train = remove(
        train,
        train["pair_key"].isin(heldout_keys),
        "train",
        "pair_overlap_with_validation_or_test",
    )
    train = remove(
        train,
        train["pair_key"].duplicated(keep="first"),
        "train",
        "duplicate_pair",
    )

    validation = original["validation"]
    validation = remove(
        validation,
        validation["pair_key"].duplicated(keep="first"),
        "validation",
        "duplicate_pair",
    )
    test = original["test"].copy()

    clean = {"train": train, "validation": validation, "test": test}

    for split, frame in clean.items():
        if not frame["pair_key"].is_unique:
            raise ValueError(f"{split}: duplicate pairs remain")
        if frame.empty:
            raise ValueError(f"{split}: empty split")

    for left, right in combinations(clean, 2):
        shared = set(clean[left]["pair_key"]) & set(clean[right]["pair_key"])
        if shared:
            raise ValueError(f"{left}/{right}: pair overlap remains")

    pd.testing.assert_frame_equal(test, original["test"])

    if len(train) < TRAIN_SIZE:
        raise ValueError("Not enough training examples")

    selected = stratified_sample(train)
    selected["cv_group"] = selected["premise"].map(premise_group)

    if len(selected) != TRAIN_SIZE:
        raise ValueError("Incorrect training subset size")
    if selected.groupby("premise")["cv_group"].nunique().gt(1).any():
        raise ValueError("Inconsistent premise groups")

    train.to_parquet(OUTPUT / "train_full.parquet", index=False)
    selected.to_parquet(OUTPUT / "train.parquet", index=False)
    validation.to_parquet(OUTPUT / "validation.parquet", index=False)
    test.to_parquet(OUTPUT / "test.parquet", index=False)

    pd.concat(removed, ignore_index=True).to_csv(REPORTS / "removed_examples.csv", index=False)

    selected[["example_id", "label", "cv_group"]].to_csv(
        REPORTS / "train_subset_ids.csv", index=False
    )

    summary = []
    for name, frame in {
        "train_full": train,
        "train_100k": selected,
        "validation": validation,
        "test": test,
    }.items():
        summary.append(
            {
                "split": name,
                "pairs": len(frame),
                **{
                    label_name: int(frame["label"].eq(label).sum())
                    for label, label_name in LABELS.items()
                },
            }
        )

    summary_frame = pd.DataFrame(summary)
    summary_frame.to_csv(REPORTS / "clean_dataset_summary.csv", index=False)

    group_sizes = selected.groupby("cv_group").size()
    group_sizes.rename("examples").to_csv(REPORTS / "cv_group_sizes.csv")

    source = json.loads((REPORTS / "source.json").read_text(encoding="utf-8"))
    id_bytes = ("\n".join(selected["example_id"]) + "\n").encode("utf-8")
    policy = {
        "source": source,
        "seed": SEED,
        "train_size": TRAIN_SIZE,
        "sampling": "Proportional stratified sampling with largest remainders",
        "train_subset_ordered_ids_sha256": hashlib.sha256(id_bytes).hexdigest(),
        "train_conflicts": "Remove every occurrence of conflicting pair keys",
        "train_overlap": "Remove pairs found in original valid validation/test",
        "duplicates": "Keep first occurrence in train and validation",
        "test": "Unchanged from prepared valid-label test",
        "cv_group": "SHA256 of casefolded normalized premise",
        "note": (
            "CV grouping is by textual premise, not original image identity. "
            "Official validation/test boundaries are preserved."
        ),
    }
    (REPORTS / "experiment_data_policy.json").write_text(
        json.dumps(policy, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nFINAL DATASET")
    print(summary_frame.to_string(index=False))
    print(f"\nTraining subset seed: {SEED}")
    print(f"CV groups: {len(group_sizes)}")
    print(f"Groups with multiple examples: {int(group_sizes.gt(1).sum())}")
    print(f"Largest CV group: {int(group_sizes.max())} examples")
    print(f"Removed rows during cleaning: {sum(len(part) for part in removed)}")
    print("\nChecks passed: unique pairs, disjoint pair keys, test unchanged.")
    print("Saved to data/clean/snli/")


if __name__ == "__main__":
    main()
