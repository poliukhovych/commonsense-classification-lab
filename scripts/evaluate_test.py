"""Evaluate frozen SNLI models on the same held-out test split."""

import gc
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
)
from threadpoolctl import threadpool_limits
from torch.utils.data import DataLoader, TensorDataset
from train_baseline import pair_features
from train_textcnn import TextCNN, encode
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts/snli"
REPORTS = ROOT / "reports/metrics/snli/test"
FIGURES = ROOT / "reports/figures/snli/test"
LABELS = ["entailment", "neutral", "contradiction"]


def release_memory() -> None:
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def save_results(name, frame, probabilities):
    probabilities = np.asarray(probabilities)
    if probabilities.shape != (len(frame), 3):
        raise ValueError(f"{name}: unexpected probability shape")
    if not np.isfinite(probabilities).all():
        raise ValueError(f"{name}: nonfinite probabilities")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError(f"{name}: probabilities do not sum to one")

    prediction = probabilities.argmax(axis=1)
    target = frame["label"].to_numpy()

    report = classification_report(
        target,
        prediction,
        labels=[0, 1, 2],
        target_names=LABELS,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(REPORTS / f"{name}_classification_report.csv")

    output = frame[["example_id", "premise", "hypothesis", "label"]].copy()
    output["prediction"] = prediction
    for index, label in enumerate(LABELS):
        output[f"prob_{label}"] = probabilities[:, index]
    output.to_parquet(ARTIFACTS / f"{name}_test_predictions.parquet", index=False)

    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(
        target,
        prediction,
        labels=[0, 1, 2],
        display_labels=LABELS,
        cmap="Blues",
        colorbar=False,
        ax=ax,
    )
    ax.set_title(f"{name} — test")
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(FIGURES / f"{name}_confusion.{extension}", dpi=200)
    plt.close(fig)

    metrics = {
        "model": name,
        "test_pairs": len(frame),
        "accuracy": accuracy_score(target, prediction),
        "macro_f1": f1_score(target, prediction, average="macro"),
        "weighted_f1": f1_score(target, prediction, average="weighted"),
        **{f"f1_{label}": report[label]["f1-score"] for label in LABELS},
    }
    (REPORTS / f"{name}_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\n{name.upper()} TEST RESULTS")
    print(
        classification_report(
            target,
            prediction,
            labels=[0, 1, 2],
            target_names=LABELS,
            digits=4,
            zero_division=0,
        )
    )
    return metrics


def baseline_probabilities(frame):
    bundle = joblib.load(ARTIFACTS / "tfidf_logreg.joblib")
    if list(bundle["classifier"].classes_) != [0, 1, 2]:
        raise ValueError("Unexpected baseline class order")
    features = pair_features(bundle["vectorizer"], frame)
    return bundle["classifier"].predict_proba(features)


def textcnn_probabilities(frame, device):
    checkpoint = torch.load(
        ARTIFACTS / "textcnn_best.pt",
        map_location="cpu",
        weights_only=True,
    )
    metadata = checkpoint["metadata"]
    vocabulary = checkpoint["vocabulary"]

    model = TextCNN(len(vocabulary), metadata["config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()

    dataset = TensorDataset(
        encode(frame["premise"], vocabulary, metadata["max_lengths"]["premise"]),
        encode(
            frame["hypothesis"],
            vocabulary,
            metadata["max_lengths"]["hypothesis"],
        ),
    )
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)
    probabilities = []

    with torch.inference_mode():
        for premise, hypothesis in loader:
            logits = model(premise.to(device), hypothesis.to(device))
            probabilities.append(logits.softmax(dim=1).cpu().numpy())

    return np.concatenate(probabilities)


def distilbert_probabilities(frame, device):
    model_path = ARTIFACTS / "distilbert_best"
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
    if [model.config.id2label[index] for index in range(3)] != LABELS:
        raise ValueError("Unexpected DistilBERT class order")
    model.to(device).eval()

    config = json.loads(
        (ROOT / "reports/metrics/snli/distilbert/config.json").read_text(encoding="utf-8")
    )
    probabilities = []
    batch_size = 32

    with torch.inference_mode():
        for start in range(0, len(frame), batch_size):
            batch = frame.iloc[start : start + batch_size]
            inputs = tokenizer(
                batch["premise"].tolist(),
                batch["hypothesis"].tolist(),
                padding=True,
                truncation="longest_first",
                max_length=config["max_length"],
                return_token_type_ids=False,
                return_tensors="pt",
            ).to(device)
            logits = model(**inputs).logits
            probabilities.append(logits.softmax(dim=1).cpu().numpy())

    return np.concatenate(probabilities)


def main():
    for directory in (REPORTS, FIGURES):
        directory.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(4)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    frame = pd.read_parquet(ROOT / "data/clean/snli/test.parquet")

    if not frame["example_id"].is_unique:
        raise ValueError("Test example IDs must be unique")
    if not frame["label"].isin([0, 1, 2]).all():
        raise ValueError("Invalid test labels")

    print(f"Test pairs: {len(frame)}")
    print(f"Neural network device: {device}")
    results = []

    print("\nEvaluating TF-IDF + Logistic Regression...", flush=True)
    probabilities = baseline_probabilities(frame)
    results.append(save_results("baseline", frame, probabilities))
    release_memory()

    print("\nEvaluating TextCNN...", flush=True)
    probabilities = textcnn_probabilities(frame, device)
    results.append(save_results("textcnn", frame, probabilities))
    release_memory()

    print("\nEvaluating DistilBERT...", flush=True)
    probabilities = distilbert_probabilities(frame, device)
    results.append(save_results("distilbert", frame, probabilities))
    release_memory()

    comparison = pd.DataFrame(results)
    comparison.to_csv(REPORTS / "comparison.csv", index=False)

    print("\nFINAL TEST COMPARISON")
    print(comparison.to_string(index=False))
    print("\nNo model was trained or tuned in this script.")


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
