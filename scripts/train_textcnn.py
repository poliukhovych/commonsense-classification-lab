"""Train a shared TextCNN pair classifier on SNLI."""

import copy
import json
import random
import re
import time
from collections import Counter
from pathlib import Path

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
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/clean/snli"
REPORTS = ROOT / "reports/metrics/snli/textcnn"
FIGURES = ROOT / "reports/figures/snli"
ARTIFACTS = ROOT / "artifacts/snli"

LABELS = ["entailment", "neutral", "contradiction"]
TOKEN_PATTERN = r"\w+|[^\w\s]"

CONFIG = {
    "seed": 42,
    "vocab_limit": 30_000,
    "min_frequency": 2,
    "embedding_dim": 128,
    "filters": 64,
    "kernels": [3, 4, 5],
    "hidden_dim": 128,
    "dropout": 0.3,
    "batch_size": 128,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "max_epochs": 15,
    "patience": 3,
    "length_quantile": 0.99,
    "length_cap": 128,
}


def tokenize(text: str) -> list[str]:
    return re.findall(TOKEN_PATTERN, text.lower())


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def encode(texts, vocabulary: dict, max_length: int) -> torch.Tensor:
    result = torch.zeros((len(texts), max_length), dtype=torch.long)
    for index, text in enumerate(texts):
        ids = [vocabulary.get(token, 1) for token in tokenize(text)[:max_length]]
        if not ids:
            ids = [1]
        result[index, : len(ids)] = torch.tensor(ids, dtype=torch.long)
    return result


class TextCNN(nn.Module):
    def __init__(self, vocabulary_size: int, config: dict):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, config["embedding_dim"], padding_idx=0)
        self.kernels = config["kernels"]
        self.convolutions = nn.ModuleList(
            [
                nn.Conv1d(config["embedding_dim"], config["filters"], kernel)
                for kernel in self.kernels
            ]
        )
        sentence_dim = config["filters"] * len(self.kernels)
        self.classifier = nn.Sequential(
            nn.Dropout(config["dropout"]),
            nn.Linear(sentence_dim * 4, config["hidden_dim"]),
            nn.ReLU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(config["hidden_dim"], 3),
        )

    def encode_sentence(self, tokens: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(tokens).transpose(1, 2)
        valid_starts = tokens.ne(0).unsqueeze(1)
        pooled = []

        for kernel, convolution in zip(self.kernels, self.convolutions, strict=True):
            # Right padding yields one window per input position.
            # Windows starting at real tokens may contain trailing padding.
            values = F.relu(convolution(F.pad(embedded, (0, kernel - 1))))
            # Exclude windows starting entirely in the padded region.
            values = values.masked_fill(~valid_starts, float("-inf"))
            pooled.append(values.amax(dim=2))

        return torch.cat(pooled, dim=1)

    def forward(self, premise: torch.Tensor, hypothesis: torch.Tensor):
        u = self.encode_sentence(premise)
        v = self.encode_sentence(hypothesis)
        pair = torch.cat([u, v, torch.abs(u - v), u * v], dim=1)
        return self.classifier(pair)


def evaluate(model, loader, device):
    model.eval()
    loss_sum = 0.0
    probabilities = []
    targets = []

    with torch.inference_mode():
        for premise, hypothesis, labels in loader:
            premise = premise.to(device)
            hypothesis = hypothesis.to(device)
            labels = labels.to(device)

            logits = model(premise, hypothesis)
            loss_sum += F.cross_entropy(logits, labels, reduction="sum").item()
            probabilities.append(logits.softmax(dim=1).cpu().numpy())
            targets.append(labels.cpu().numpy())

    probabilities = np.concatenate(probabilities)
    targets = np.concatenate(targets)
    predictions = probabilities.argmax(axis=1)

    return {
        "loss": loss_sum / len(targets),
        "accuracy": accuracy_score(targets, predictions),
        "macro_f1": f1_score(targets, predictions, average="macro"),
    }, probabilities


def main() -> None:
    for directory in (REPORTS, FIGURES, ARTIFACTS):
        directory.mkdir(parents=True, exist_ok=True)

    seed = CONFIG["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    train = pd.read_parquet(DATA / "train.parquet")
    validation = pd.read_parquet(DATA / "validation.parquet")

    counter = Counter()
    lengths = {}
    for column in ("premise", "hypothesis"):
        column_lengths = []
        for text in train[column]:
            tokens = tokenize(text)
            counter.update(tokens)
            column_lengths.append(len(tokens))
        lengths[column] = np.asarray(column_lengths)

    vocabulary = {"<PAD>": 0, "<UNK>": 1}
    for token, count in counter.most_common():
        if len(vocabulary) >= CONFIG["vocab_limit"]:
            break
        if count >= CONFIG["min_frequency"] and token not in vocabulary:
            vocabulary[token] = len(vocabulary)

    max_lengths = {
        column: max(
            max(CONFIG["kernels"]),
            min(
                CONFIG["length_cap"],
                int(np.quantile(values, CONFIG["length_quantile"], method="higher")),
            ),
        )
        for column, values in lengths.items()
    }
    truncation = {
        column: float((values > max_lengths[column]).mean()) for column, values in lengths.items()
    }

    def make_dataset(frame):
        return TensorDataset(
            encode(frame["premise"], vocabulary, max_lengths["premise"]),
            encode(frame["hypothesis"], vocabulary, max_lengths["hypothesis"]),
            torch.tensor(frame["label"].to_numpy(), dtype=torch.long),
        )

    train_dataset = make_dataset(train)
    validation_dataset = make_dataset(validation)
    generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=0,
    )

    model = TextCNN(len(vocabulary), CONFIG).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    metadata = {
        "config": CONFIG,
        "vocabulary_size": len(vocabulary),
        "max_lengths": max_lengths,
        "train_truncation_fraction": truncation,
        "token_pattern": TOKEN_PATTERN,
        "lowercase": True,
        "parameter_count": parameter_count,
        "device": str(device),
        "torch_version": str(torch.__version__),
        "train_pairs": len(train),
        "validation_pairs": len(validation),
    }
    (REPORTS / "config.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"Vocabulary: {len(vocabulary)}", flush=True)
    print(f"Max lengths: {max_lengths}", flush=True)
    print(f"Train truncation fractions: {truncation}", flush=True)
    print(f"Trainable parameters: {parameter_count:,}", flush=True)

    best_score = -1.0
    best_epoch = 0
    stale_epochs = 0
    history = []
    checkpoint_path = ARTIFACTS / "textcnn_best.pt"

    synchronize(device)
    training_start = time.perf_counter()

    for epoch in range(1, CONFIG["max_epochs"] + 1):
        epoch_start = time.perf_counter()
        model.train()
        running_loss = 0.0
        seen = 0

        for step, (premise, hypothesis, labels) in enumerate(train_loader, start=1):
            premise = premise.to(device)
            hypothesis = hypothesis.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(premise, hypothesis)
            loss = F.cross_entropy(logits, labels)

            if not torch.isfinite(loss).item():
                raise RuntimeError("Nonfinite training loss")

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * len(labels)
            seen += len(labels)

            if step % 200 == 0:
                print(
                    f"Epoch {epoch}: batch {step}/{len(train_loader)}, "
                    f"loss={running_loss / seen:.4f}",
                    flush=True,
                )

        validation_metrics, _ = evaluate(model, validation_loader, device)
        synchronize(device)
        epoch_seconds = time.perf_counter() - epoch_start

        row = {
            "epoch": epoch,
            "train_loss": running_loss / seen,
            "validation_loss": validation_metrics["loss"],
            "validation_accuracy": validation_metrics["accuracy"],
            "validation_macro_f1": validation_metrics["macro_f1"],
            "epoch_seconds": epoch_seconds,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(REPORTS / "history.csv", index=False)

        score = validation_metrics["macro_f1"]
        print(
            f"Epoch {epoch:02d}: train_loss={row['train_loss']:.4f}, "
            f"val_loss={row['validation_loss']:.4f}, "
            f"val_F1={score:.4f}, time={epoch_seconds:.1f}s",
            flush=True,
        )

        if score > best_score:
            best_score = score
            best_epoch = epoch
            stale_epochs = 0
            state = {
                name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()
            }
            torch.save(
                {
                    "state_dict": state,
                    "vocabulary": vocabulary,
                    "metadata": copy.deepcopy(metadata),
                    "best_epoch": best_epoch,
                    "validation_macro_f1": best_score,
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= CONFIG["patience"]:
                print("Early stopping.", flush=True)
                break

    synchronize(device)
    training_seconds = time.perf_counter() - training_start

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])

    final_metrics, probabilities = evaluate(model, validation_loader, device)
    predictions = probabilities.argmax(axis=1)
    y_validation = validation["label"].to_numpy()

    report = classification_report(
        y_validation,
        predictions,
        labels=[0, 1, 2],
        target_names=LABELS,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(REPORTS / "validation_classification_report.csv")

    output = validation[["example_id", "premise", "hypothesis", "label"]].copy()
    output["prediction"] = predictions
    for index, name in enumerate(LABELS):
        output[f"prob_{name}"] = probabilities[:, index]
    output.to_parquet(ARTIFACTS / "textcnn_validation_predictions.parquet", index=False)

    metrics = {
        **final_metrics,
        "weighted_f1": f1_score(y_validation, predictions, average="weighted"),
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "training_including_validation_and_checkpoints_seconds": training_seconds,
        "parameter_count": parameter_count,
    }
    (REPORTS / "validation_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )

    history_frame = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(history_frame["epoch"], history_frame["train_loss"], label="Train")
    axes[0].plot(
        history_frame["epoch"],
        history_frame["validation_loss"],
        label="Validation",
    )
    axes[0].set(xlabel="Epoch", ylabel="Cross-entropy", title="Loss")
    axes[0].legend()
    axes[1].plot(
        history_frame["epoch"],
        history_frame["validation_macro_f1"],
        marker="o",
    )
    axes[1].axvline(best_epoch, color="gray", linestyle="--", label="Best epoch")
    axes[1].set(xlabel="Epoch", ylabel="Macro-F1", title="Validation Macro-F1")
    axes[1].legend()
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(FIGURES / f"textcnn_learning_curves.{extension}", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_validation,
        predictions,
        labels=[0, 1, 2],
        display_labels=LABELS,
        cmap="Blues",
        colorbar=False,
        ax=ax,
    )
    ax.set_title("TextCNN — validation")
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(FIGURES / f"textcnn_confusion_matrix.{extension}", dpi=200)
    plt.close(fig)

    print("\nTEXTCNN VALIDATION RESULTS")
    print(
        classification_report(
            y_validation,
            predictions,
            labels=[0, 1, 2],
            target_names=LABELS,
            digits=4,
            zero_division=0,
        )
    )
    print(f"Best epoch: {best_epoch}")
    print(f"Epochs run: {len(history)}")
    print(f"Best checkpoint Macro-F1: {final_metrics['macro_f1']:.4f}")
    print(f"Training loop including validation: {training_seconds:.1f}s")
    print(f"Checkpoint: {checkpoint_path}")
    print("Test split was not loaded.")


if __name__ == "__main__":
    main()
