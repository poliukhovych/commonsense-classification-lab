"""Build report figures and error-analysis tables from saved predictions."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/clean/snli"
ARTIFACTS = ROOT / "artifacts/snli"
METRICS = ROOT / "reports/metrics/snli"
FIGURES = ROOT / "reports/figures/snli/report"
TABLES = ROOT / "reports/tables/snli"

LABELS = ["entailment", "neutral", "contradiction"]
MODELS = ["baseline", "textcnn", "distilbert"]
NAMES = {
    "baseline": "TF-IDF + LR",
    "textcnn": "TextCNN",
    "distilbert": "DistilBERT",
}
COLORS = ["#4C78A8", "#F28E2B", "#59A14F"]
SEED = 42


def save_figure(fig, name):
    for extension in ("png", "pdf"):
        fig.savefig(
            FIGURES / f"{name}.{extension}",
            dpi=220,
            bbox_inches="tight",
        )
    plt.close(fig)


def annotate_bars(ax, bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f}",
            (bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
        }
    )

    test = pd.read_parquet(DATA / "test.parquet")
    if not test["example_id"].is_unique:
        raise ValueError("Duplicate test IDs")

    test_ids = pd.Index(test["example_id"])
    target = test["label"].to_numpy()
    joined = test[["example_id", "premise", "hypothesis", "label"]].copy()
    joined["true_label"] = joined["label"].map(dict(enumerate(LABELS)))

    comparison = pd.read_csv(METRICS / "test/comparison.csv")
    comparison = comparison.set_index("model").loc[MODELS].reset_index()
    predictions = {}

    for model in MODELS:
        frame = pd.read_parquet(ARTIFACTS / f"{model}_test_predictions.parquet")
        if not frame["example_id"].is_unique:
            raise ValueError(f"{model}: duplicate prediction IDs")
        if set(frame["example_id"]) != set(test_ids):
            raise ValueError(f"{model}: test IDs do not match")

        frame = frame.set_index("example_id").loc[test_ids].reset_index()
        if not np.array_equal(frame["label"].to_numpy(), target):
            raise ValueError(f"{model}: labels do not match")

        for column in ("premise", "hypothesis"):
            if not np.array_equal(frame[column].to_numpy(), test[column].to_numpy()):
                raise ValueError(f"{model}: text mismatch")

        predicted = frame["prediction"].to_numpy()
        if not np.isin(predicted, [0, 1, 2]).all():
            raise ValueError(f"{model}: invalid predictions")
        predictions[model] = predicted

        recorded = comparison.loc[comparison["model"].eq(model)].iloc[0]
        actual_accuracy = accuracy_score(target, predicted)
        actual_f1 = f1_score(target, predicted, average="macro")
        if not np.isclose(actual_accuracy, recorded["accuracy"]):
            raise ValueError(f"{model}: recorded accuracy mismatch")
        if not np.isclose(actual_f1, recorded["macro_f1"]):
            raise ValueError(f"{model}: recorded F1 mismatch")

        joined[f"{model}_prediction"] = [LABELS[int(label)] for label in predicted]
        probability_columns = [f"prob_{label}" for label in LABELS]
        joined[f"{model}_confidence"] = frame[probability_columns].max(axis=1).to_numpy()
        joined[f"{model}_correct"] = predicted == target

    comparison.to_csv(TABLES / "test_comparison.csv", index=False)

    # Overall test metrics.
    positions = np.arange(len(MODELS))
    width = 0.34
    fig, ax = plt.subplots(figsize=(9, 5))
    for offset, column, label, color in (
        (-width / 2, "accuracy", "Accuracy", "#4C78A8"),
        (width / 2, "macro_f1", "Macro-F1", "#59A14F"),
    ):
        bars = ax.bar(
            positions + offset,
            comparison[column].to_numpy() * 100,
            width,
            label=label,
            color=color,
        )
        annotate_bars(ax, bars)

    ax.set_xticks(positions, [NAMES[model] for model in MODELS])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Score (%)")
    ax.set_title("Model comparison on SNLI test")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, "test_overall_metrics")

    # Per-class F1.
    positions = np.arange(len(LABELS))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10, 5))
    for index, model in enumerate(MODELS):
        row = comparison.loc[comparison["model"].eq(model)].iloc[0]
        values = [row[f"f1_{label}"] * 100 for label in LABELS]
        bars = ax.bar(
            positions + (index - 1) * width,
            values,
            width,
            label=NAMES[model],
            color=COLORS[index],
        )
        annotate_bars(ax, bars)

    ax.set_xticks(positions, LABELS)
    ax.set_ylim(0, 100)
    ax.set_ylabel("F1 (%)")
    ax.set_title("Test F1 by class")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, "test_f1_by_class")

    # Shared color scale; each row sums to 100%.
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), layout="constrained")
    for ax, model in zip(axes, MODELS, strict=True):
        matrix = (
            confusion_matrix(target, predictions[model], labels=[0, 1, 2], normalize="true") * 100
        )
        image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=100)

        for row in range(3):
            for column in range(3):
                value = matrix[row, column]
                ax.text(
                    column,
                    row,
                    f"{value:.1f}%",
                    ha="center",
                    va="center",
                    color="white" if value > 55 else "black",
                )

        ax.set_xticks(range(3), LABELS, rotation=25, ha="right")
        ax.set_yticks(range(3), LABELS)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.set_title(NAMES[model])

    fig.colorbar(image, ax=list(axes), label="Share of true class (%)", shrink=0.8)
    save_figure(fig, "test_confusion_matrices_normalized")

    # Class distributions in actual experimental splits.
    split_frames = {
        split: pd.read_parquet(DATA / f"{split}.parquet")
        for split in ("train", "validation", "test")
    }
    distribution_rows = []
    length_rows = []

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, (split, frame) in zip(axes, split_frames.items(), strict=True):
        counts = frame["label"].value_counts().reindex([0, 1, 2], fill_value=0)
        percentages = counts.to_numpy() / len(frame) * 100
        bars = ax.bar(LABELS, percentages, color=COLORS)
        ax.set_ylim(0, 45)
        ax.set_title(f"{split.capitalize()} (n={len(frame):,})")
        ax.set_ylabel("Share (%)")
        ax.tick_params(axis="x", rotation=20)

        for index, bar in enumerate(bars):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.8,
                f"{int(counts.iloc[index]):,}\n({percentages[index]:.1f}%)",
                ha="center",
                fontsize=9,
            )

        for label, count in counts.items():
            distribution_rows.append(
                {
                    "split": split,
                    "class": LABELS[int(label)],
                    "count": int(count),
                    "percentage": float(count / len(frame) * 100),
                }
            )

        for field in ("premise", "hypothesis"):
            lengths = frame[field].str.split().str.len()
            length_rows.append(
                {
                    "split": split,
                    "field": field,
                    "mean": lengths.mean(),
                    "median": lengths.median(),
                    "p95": lengths.quantile(0.95),
                    "p99": lengths.quantile(0.99),
                    "max": lengths.max(),
                }
            )

    fig.tight_layout()
    save_figure(fig, "class_distribution")
    pd.DataFrame(distribution_rows).to_csv(TABLES / "class_distribution.csv", index=False)
    pd.DataFrame(length_rows).to_csv(TABLES / "text_lengths.csv", index=False)

    # Descriptive lengths, measured in whitespace-separated words.
    train = split_frames["train"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, field, color in zip(axes, ("premise", "hypothesis"), COLORS[:2], strict=True):
        lengths = train[field].str.split().str.len()
        bins = np.arange(0.5, int(lengths.max()) + 1.5, 1)
        ax.hist(lengths, bins=bins, color=color, edgecolor="white")
        ax.set(
            xlabel="Whitespace-separated words",
            ylabel="Training examples",
            title=field.capitalize(),
        )
    fig.tight_layout()
    save_figure(fig, "train_text_lengths")

    # Data preparation counts from existing audit files.
    raw = pd.read_csv(METRICS / "dataset_summary.csv").set_index("split")
    clean = pd.read_csv(METRICS / "clean_dataset_summary.csv").set_index("split")
    preparation_rows = []

    for split in ("train", "validation", "test"):
        clean_name = "train_full" if split == "train" else split
        valid_count = int(raw.loc[split, "valid_pairs"])
        clean_count = int(clean.loc[clean_name, "pairs"])
        preparation_rows.append(
            {
                "split": split,
                "raw_pairs": int(raw.loc[split, "raw_pairs"]),
                "valid_pairs": valid_count,
                "removed_by_cleaning": valid_count - clean_count,
                "clean_pairs": clean_count,
                "experiment_pairs": len(split_frames[split]),
                "not_sampled_from_clean_train": (
                    clean_count - len(split_frames[split]) if split == "train" else 0
                ),
            }
        )

    pd.DataFrame(preparation_rows).to_csv(TABLES / "data_preparation.csv", index=False)

    # Error analysis: select reproducibly, not by a hand-picked narrative.
    all_wrong = np.logical_and.reduce([~joined[f"{model}_correct"].to_numpy() for model in MODELS])
    transformer_rescues = (
        joined["distilbert_correct"] & ~joined["baseline_correct"] & ~joined["textcnn_correct"]
    )
    simpler_models_win = (
        ~joined["distilbert_correct"] & joined["baseline_correct"] & joined["textcnn_correct"]
    )

    categories = {
        "all_models_wrong": joined.loc[all_wrong],
        "distilbert_correct_others_wrong": joined.loc[transformer_rescues],
        "simpler_models_correct_distilbert_wrong": joined.loc[simpler_models_win],
    }

    category_counts = []
    for name, frame in categories.items():
        category_counts.append({"category": name, "examples": len(frame)})
        frame.to_csv(TABLES / f"{name}_all.csv", index=False)

        sample = frame.sample(n=min(12, len(frame)), random_state=SEED)
        sample.to_csv(TABLES / f"{name}_sample.csv", index=False)

    counts_frame = pd.DataFrame(category_counts)
    counts_frame.to_csv(TABLES / "error_categories.csv", index=False)

    confusion_rows = []
    for model in MODELS:
        matrix = confusion_matrix(target, predictions[model], labels=[0, 1, 2])
        for true_label in range(3):
            for predicted_label in range(3):
                if true_label != predicted_label:
                    confusion_rows.append(
                        {
                            "model": model,
                            "true_class": LABELS[true_label],
                            "predicted_class": LABELS[predicted_label],
                            "count": int(matrix[true_label, predicted_label]),
                        }
                    )
    pd.DataFrame(confusion_rows).sort_values(["model", "count"], ascending=[True, False]).to_csv(
        TABLES / "confusion_pairs.csv", index=False
    )

    print("\nREPORT MATERIALS CREATED")
    print(f"Figures: {FIGURES}")
    print(f"Tables: {TABLES}")
    print("\nDATA PREPARATION")
    print(pd.DataFrame(preparation_rows).to_string(index=False))
    print("\nERROR CATEGORIES")
    print(counts_frame.to_string(index=False))

    print("\nSAMPLE: ALL MODELS WRONG")
    sample = categories["all_models_wrong"].sample(
        n=min(3, len(categories["all_models_wrong"])),
        random_state=SEED,
    )
    for row in sample.itertuples(index=False):
        print(f"\nID: {row.example_id}")
        print(f"Premise: {row.premise}")
        print(f"Hypothesis: {row.hypothesis}")
        print(f"Dataset label: {row.true_label}")
        print(
            f"Predictions: LR={row.baseline_prediction}, "
            f"CNN={row.textcnn_prediction}, "
            f"DistilBERT={row.distilbert_prediction}"
        )

    print("\nPrediction alignment and recorded metric checks passed.")


if __name__ == "__main__":
    main()
