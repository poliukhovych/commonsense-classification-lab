"""Train a TF-IDF pair classifier with grouped CV; evaluate validation only."""

import json
import platform
import time
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from scipy.sparse import hstack
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/clean/snli"
REPORTS = ROOT / "reports/metrics/snli/baseline"
FIGURES = ROOT / "reports/figures/snli"
ARTIFACTS = ROOT / "artifacts/snli"

LABELS = ["entailment", "neutral", "contradiction"]
C_VALUES = [0.1, 1.0, 10.0]
SEED = 42
MAX_FEATURES = 20_000


def fit_vectorizer(frame: pd.DataFrame) -> TfidfVectorizer:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_features=MAX_FEATURES,
        sublinear_tf=True,
        dtype=np.float64,
    )
    documents = pd.concat([frame["premise"], frame["hypothesis"]], ignore_index=True)
    vectorizer.fit(documents)
    return vectorizer


def pair_features(vectorizer: TfidfVectorizer, frame: pd.DataFrame):
    premise = vectorizer.transform(frame["premise"])
    hypothesis = vectorizer.transform(frame["hypothesis"])

    difference = (premise - hypothesis).tocsr()
    difference.data = np.abs(difference.data)
    difference.eliminate_zeros()

    return hstack(
        [premise, hypothesis, difference, premise.multiply(hypothesis)],
        format="csr",
    )


def fit_classifier(features, labels, c_value: float):
    model = LogisticRegression(
        C=c_value,
        solver="lbfgs",
        max_iter=1000,
        tol=1e-4,
        random_state=SEED,
    )
    start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(features, labels)
    return model, time.perf_counter() - start


def main() -> None:
    for directory in (REPORTS, FIGURES, ARTIFACTS):
        directory.mkdir(parents=True, exist_ok=True)

    train = pd.read_parquet(DATA / "train.parquet")
    validation = pd.read_parquet(DATA / "validation.parquet")

    y = train["label"].to_numpy()
    groups = train["cv_group"].to_numpy()
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)

    results = []
    fold_assignment = np.full(len(train), -1, dtype=int)
    search_start = time.perf_counter()

    for fold, (fit_indices, check_indices) in enumerate(cv.split(train, y, groups), start=1):
        if set(groups[fit_indices]) & set(groups[check_indices]):
            raise ValueError("CV group leakage detected")

        fold_assignment[check_indices] = fold
        fit_frame = train.iloc[fit_indices]
        check_frame = train.iloc[check_indices]

        print(f"\nFold {fold}/5: building TF-IDF features...", flush=True)
        vectorizer = fit_vectorizer(fit_frame)
        x_fit = pair_features(vectorizer, fit_frame)
        x_check = pair_features(vectorizer, check_frame)

        for c_value in C_VALUES:
            print(f"  Training C={c_value}...", flush=True)
            model, fit_seconds = fit_classifier(x_fit, y[fit_indices], c_value)
            prediction = model.predict(x_check)
            macro_f1 = f1_score(y[check_indices], prediction, average="macro")
            results.append(
                {
                    "fold": fold,
                    "C": c_value,
                    "macro_f1": macro_f1,
                    "accuracy": accuracy_score(y[check_indices], prediction),
                    "classifier_fit_seconds": fit_seconds,
                    "iterations": int(model.n_iter_.max()),
                }
            )
            pd.DataFrame(results).to_csv(REPORTS / "cv_results.csv", index=False)
            print(
                f"  Macro-F1={macro_f1:.4f}; fit={fit_seconds:.1f}s",
                flush=True,
            )

        del x_fit, x_check, vectorizer, model

    search_seconds = time.perf_counter() - search_start

    if (fold_assignment < 0).any():
        raise ValueError("Some examples have no CV fold")

    assignments = train[["example_id", "cv_group"]].copy()
    assignments["fold"] = fold_assignment
    assignments.to_csv(REPORTS / "cv_folds.csv", index=False)

    cv_summary = (
        pd.DataFrame(results)
        .groupby("C", as_index=False)
        .agg(
            mean_macro_f1=("macro_f1", "mean"),
            std_macro_f1=("macro_f1", "std"),
            mean_accuracy=("accuracy", "mean"),
        )
        .sort_values(
            ["mean_macro_f1", "C"],
            ascending=[False, True],
        )
    )
    cv_summary.to_csv(REPORTS / "cv_summary.csv", index=False)
    best_c = float(cv_summary.iloc[0]["C"])

    print("\nCV SUMMARY")
    print(cv_summary.to_string(index=False))
    print(f"\nRefitting on all training pairs with C={best_c}...", flush=True)

    refit_start = time.perf_counter()
    vectorizer = fit_vectorizer(train)
    x_train = pair_features(vectorizer, train)
    model, classifier_seconds = fit_classifier(x_train, y, best_c)
    refit_seconds = time.perf_counter() - refit_start

    inference_start = time.perf_counter()
    x_validation = pair_features(vectorizer, validation)
    probabilities = model.predict_proba(x_validation)
    prediction = model.classes_[probabilities.argmax(axis=1)]
    inference_seconds = time.perf_counter() - inference_start

    y_validation = validation["label"].to_numpy()
    report = classification_report(
        y_validation,
        prediction,
        labels=[0, 1, 2],
        target_names=LABELS,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(REPORTS / "validation_classification_report.csv")

    predictions = validation[["example_id", "premise", "hypothesis", "label"]].copy()
    predictions["prediction"] = prediction
    for index, label in enumerate(model.classes_):
        predictions[f"prob_{LABELS[int(label)]}"] = probabilities[:, index]
    predictions.to_parquet(ARTIFACTS / "baseline_validation_predictions.parquet", index=False)

    majority_label = int(train["label"].value_counts().idxmax())
    majority_prediction = np.full(len(validation), majority_label)

    metrics = {
        "validation_accuracy": accuracy_score(y_validation, prediction),
        "validation_macro_f1": f1_score(y_validation, prediction, average="macro"),
        "validation_weighted_f1": f1_score(y_validation, prediction, average="weighted"),
        "majority_baseline_accuracy": accuracy_score(y_validation, majority_prediction),
        "majority_baseline_macro_f1": f1_score(y_validation, majority_prediction, average="macro"),
        "best_C": best_c,
        "cv_search_seconds": search_seconds,
        "final_fit_including_features_seconds": refit_seconds,
        "final_classifier_fit_seconds": classifier_seconds,
        "validation_batch_inference_seconds": inference_seconds,
        "validation_pairs_per_second": len(validation) / inference_seconds,
        "vocabulary_size": len(vectorizer.vocabulary_),
        "pair_feature_count": x_train.shape[1],
        "final_iterations": int(model.n_iter_.max()),
        "train_pairs": len(train),
        "validation_pairs": len(validation),
        "seed": SEED,
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "thread_limit": 4,
        "feature_blocks": ["premise", "hypothesis", "absolute_difference", "product"],
        "tfidf_max_features": MAX_FEATURES,
        "tfidf_ngram_range": [1, 2],
        "tfidf_min_df": 2,
        "tfidf_sublinear_tf": True,
    }
    (REPORTS / "validation_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )

    joblib.dump(
        {
            "vectorizer": vectorizer,
            "classifier": model,
            "labels": LABELS,
            "feature_blocks": metrics["feature_blocks"],
        },
        ARTIFACTS / "tfidf_logreg.joblib",
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_validation,
        prediction,
        labels=[0, 1, 2],
        display_labels=LABELS,
        cmap="Blues",
        colorbar=False,
        ax=ax,
    )
    ax.set_title("TF-IDF + Logistic Regression — validation")
    fig.tight_layout()
    fig.savefig(FIGURES / "baseline_confusion_matrix.png", dpi=200)
    fig.savefig(FIGURES / "baseline_confusion_matrix.pdf")
    plt.close(fig)

    print("\nVALIDATION RESULTS")
    print(
        classification_report(
            y_validation,
            prediction,
            labels=[0, 1, 2],
            target_names=LABELS,
            digits=4,
            zero_division=0,
        )
    )
    print(f"Best C: {best_c}")
    print(f"CV search: {search_seconds:.1f}s")
    print(f"Final fit including features: {refit_seconds:.1f}s")
    print(f"Majority baseline accuracy: {metrics['majority_baseline_accuracy']:.4f}")
    print("Saved model, CV results, validation predictions and confusion matrix.")
    print("Test split was not loaded.")


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
