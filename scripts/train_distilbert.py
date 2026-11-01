"""Fine-tune base DistilBERT on SNLI; evaluate validation only."""

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import transformers
from datasets import Dataset
from huggingface_hub import HfApi
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/clean/snli"
REPORTS = ROOT / "reports/metrics/snli/distilbert"
FIGURES = ROOT / "reports/figures/snli"
ARTIFACTS = ROOT / "artifacts/snli"
CHECKPOINTS = ARTIFACTS / "distilbert_checkpoints"
BEST_MODEL = ARTIFACTS / "distilbert_best"
CACHE = ROOT / "data/raw/hf-model-cache"

MODEL_ID = "distilbert/distilbert-base-uncased"
LABELS = ["entailment", "neutral", "contradiction"]
SEED = 42
MAX_LENGTH = 128


def compute_metrics(result):
    prediction = np.argmax(result.predictions, axis=1)
    return {
        "accuracy": accuracy_score(result.label_ids, prediction),
        "macro_f1": f1_score(result.label_ids, prediction, average="macro"),
        "weighted_f1": f1_score(result.label_ids, prediction, average="weighted"),
    }


def main() -> None:
    for directory in (REPORTS, FIGURES, ARTIFACTS, CACHE):
        directory.mkdir(parents=True, exist_ok=True)

    if CHECKPOINTS.exists() and any(CHECKPOINTS.glob("checkpoint-*")):
        raise RuntimeError(
            "Existing checkpoints found. Preserve or move that directory "
            "before starting a new experiment."
        )

    set_seed(SEED)
    torch.set_num_threads(4)

    source_path = REPORTS / "model_source.json"
    if source_path.exists():
        source = json.loads(source_path.read_text(encoding="utf-8"))
        if source["model_id"] != MODEL_ID:
            raise ValueError("Unexpected model source")
        revision = source["revision"]
    else:
        revision = HfApi().model_info(MODEL_ID).sha
        if not revision:
            raise ValueError("Could not resolve model revision")
        source = {"model_id": MODEL_ID, "revision": revision}
        source_path.write_text(json.dumps(source, indent=2) + "\n", encoding="utf-8")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=revision,
        cache_dir=str(CACHE),
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        revision=revision,
        cache_dir=str(CACHE),
        num_labels=3,
        id2label=dict(enumerate(LABELS)),
        label2id={name: index for index, name in enumerate(LABELS)},
    )

    train_frame = pd.read_parquet(DATA / "train.parquet")
    validation_frame = pd.read_parquet(DATA / "validation.parquet")

    def tokenize_batch(batch):
        encoded = tokenizer(
            batch["premise"],
            batch["hypothesis"],
            truncation="longest_first",
            max_length=MAX_LENGTH,
            return_token_type_ids=False,
        )
        encoded["labels"] = batch["label"]
        return encoded

    def make_dataset(frame):
        raw = Dataset.from_pandas(
            frame[["premise", "hypothesis", "label"]],
            preserve_index=False,
        )
        return raw.map(
            tokenize_batch,
            batched=True,
            remove_columns=raw.column_names,
        )

    train_dataset = make_dataset(train_frame)
    validation_dataset = make_dataset(validation_frame)

    args = TrainingArguments(
        output_dir=str(CHECKPOINTS),
        num_train_epochs=3,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=469,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=2,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        optim="adamw_torch",
        fp16=False,
        bf16=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        seed=SEED,
        data_seed=SEED,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    config = {
        **source,
        "max_length": MAX_LENGTH,
        "train_pairs": len(train_frame),
        "validation_pairs": len(validation_frame),
        "trainable_parameters": parameter_count,
        "torch_version": str(torch.__version__),
        "transformers_version": transformers.__version__,
        "device": str(args.device),
        "training_arguments": args.to_dict(),
    }
    (REPORTS / "config.json").write_text(
        json.dumps(config, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    print(f"\nDevice: {args.device}", flush=True)
    print(f"Trainable parameters: {parameter_count:,}", flush=True)
    print(f"Model revision: {revision}", flush=True)
    print("Starting fine-tuning...", flush=True)

    start = time.perf_counter()
    train_result = trainer.train()
    if args.device.type == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - start

    trainer.save_model(str(BEST_MODEL))
    tokenizer.save_pretrained(str(BEST_MODEL))

    history = pd.DataFrame(trainer.state.log_history)
    history.to_csv(REPORTS / "history.csv", index=False)
    trainer.save_state()

    result = trainer.predict(validation_dataset, metric_key_prefix="validation")
    logits = result.predictions
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    predictions = probabilities.argmax(axis=1)

    y_validation = validation_frame["label"].to_numpy()
    report = classification_report(
        y_validation,
        predictions,
        labels=[0, 1, 2],
        target_names=LABELS,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(REPORTS / "validation_classification_report.csv")

    output = validation_frame[["example_id", "premise", "hypothesis", "label"]].copy()
    output["prediction"] = predictions
    for index, name in enumerate(LABELS):
        output[f"prob_{name}"] = probabilities[:, index]
    output.to_parquet(
        ARTIFACTS / "distilbert_validation_predictions.parquet",
        index=False,
    )

    metrics = {
        **result.metrics,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_validation_macro_f1": trainer.state.best_metric,
        "training_including_validation_and_checkpoints_seconds": elapsed,
        "trainable_parameters": parameter_count,
        "train_result": train_result.metrics,
    }
    (REPORTS / "validation_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )

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
    ax.set_title("DistilBERT — validation")
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(FIGURES / f"distilbert_confusion_matrix.{extension}", dpi=200)
    plt.close(fig)

    training_logs = history.dropna(subset=["loss"])
    evaluation_logs = history.dropna(subset=["eval_loss"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(training_logs["epoch"], training_logs["loss"], label="Train (logged)")
    axes[0].plot(
        evaluation_logs["epoch"],
        evaluation_logs["eval_loss"],
        marker="o",
        label="Validation",
    )
    axes[0].set(xlabel="Epoch", ylabel="Cross-entropy", title="Loss")
    axes[0].legend()
    axes[1].plot(
        evaluation_logs["epoch"],
        evaluation_logs["eval_macro_f1"],
        marker="o",
    )
    axes[1].set(xlabel="Epoch", ylabel="Macro-F1", title="Validation Macro-F1")
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(FIGURES / f"distilbert_learning_curves.{extension}", dpi=200)
    plt.close(fig)

    print("\nDISTILBERT VALIDATION RESULTS")
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
    print(f"Best checkpoint: {trainer.state.best_model_checkpoint}")
    print(f"Training including validation: {elapsed:.1f}s")
    print(f"Best model saved to: {BEST_MODEL}")
    print("Test split was not loaded.")


if __name__ == "__main__":
    main()
