"""Post-hoc SNLI diagnostics; preserve all original experiment outputs.

Run from the repository: uv run python scripts/additional_checks.py --hypothesis-only
Without --hypothesis-only, only the three existing prediction files are analysed.
Bootstrap intervals condition on fitted models and resample normalized-premise
groups, not training seeds or original image IDs. McNemar p-values are exploratory
because example-level independence may fail. No existing model is retrained.
"""

import argparse
import json
import unicodedata
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from threadpoolctl import threadpool_limits


def normalized(text):
    return " ".join(unicodedata.normalize("NFC", str(text)).split()).casefold()


def vectorizer():
    return TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_features=20_000,
        sublinear_tf=True,
        dtype=np.float64,
    )


def classifier(c):
    return LogisticRegression(C=c, solver="lbfgs", max_iter=1000, tol=1e-4, random_state=42)


def load_test(root):
    frame = pd.read_parquet(root / "data/clean/snli/test.parquet")
    if frame.example_id.duplicated().any() or frame.empty:
        raise ValueError("Test IDs must be unique and nonempty")
    if not frame.label.isin([0, 1, 2]).all():
        raise ValueError("Invalid test labels")
    return frame.reset_index(drop=True)


def align_predictions(test, frame):
    if frame.example_id.duplicated().any():
        raise ValueError("Duplicated prediction IDs")
    if set(frame.example_id) != set(test.example_id):
        raise ValueError("Prediction/test ID sets differ")
    frame = frame.set_index("example_id").loc[test.example_id].reset_index()
    for col in ["label", "premise", "hypothesis"]:
        if not np.array_equal(frame[col].to_numpy(), test[col].to_numpy()):
            raise ValueError(f"Prediction alignment failed: {col}")
    if not frame.prediction.isin([0, 1, 2]).all():
        raise ValueError("Invalid predictions")
    return frame.prediction.to_numpy(dtype=int)


def hypothesis_only(root, out):
    """Select C on the SAME saved training folds; load test only after selection."""
    train = pd.read_parquet(root / "data/clean/snli/train.parquet")
    folds = pd.read_csv(root / "reports/metrics/snli/baseline/cv_folds.csv")
    if train.example_id.duplicated().any() or folds.example_id.duplicated().any():
        raise ValueError("Duplicated training/fold IDs")
    if set(train.example_id) != set(folds.example_id):
        raise ValueError("Training/fold ID sets differ")
    folds = folds.set_index("example_id").loc[train.example_id]
    assignments = folds["fold"].to_numpy()
    if set(assignments) != {1, 2, 3, 4, 5}:
        raise ValueError("Expected five saved folds")
    if not np.array_equal(train.cv_group.to_numpy(), folds.cv_group.to_numpy()):
        raise ValueError("Training CV group mismatch")
    y = train.label.to_numpy(dtype=int)
    rows = []
    for fold in range(1, 6):
        fit, check = assignments != fold, assignments == fold
        if set(train.loc[fit, "cv_group"]) & set(train.loc[check, "cv_group"]):
            raise ValueError("CV group leakage")
        tfidf = vectorizer()
        x_fit = tfidf.fit_transform(train.loc[fit, "hypothesis"])
        x_check = tfidf.transform(train.loc[check, "hypothesis"])
        for c in [0.1, 1.0, 10.0]:
            model = classifier(c).fit(x_fit, y[fit])
            prediction = model.predict(x_check)
            score = f1_score(
                y[check], prediction, labels=[0, 1, 2], average="macro", zero_division=0
            )
            rows.append({"fold": fold, "C": c, "macro_f1": score})
            print(f"Hypothesis-only fold={fold}, C={c}: Macro-F1={score:.4f}", flush=True)
    results = pd.DataFrame(rows)
    summary = (
        results.groupby("C", as_index=False)
        .agg(mean_macro_f1=("macro_f1", "mean"), std_macro_f1=("macro_f1", "std"))
        .sort_values(["mean_macro_f1", "C"], ascending=[False, True])
    )
    results.to_csv(out / "hypothesis_only_cv.csv", index=False)
    summary.to_csv(out / "hypothesis_only_cv_summary.csv", index=False)
    best = float(summary.iloc[0].C)
    tfidf = vectorizer()
    model = classifier(best).fit(tfidf.fit_transform(train.hypothesis), y)
    metrics = {
        "best_C": best,
        "selection": "saved five grouped training folds",
        "input": "hypothesis only",
        "train_pairs": len(train),
        "post_hoc_diagnostic": True,
    }
    for split in ["validation", "test"]:
        frame = pd.read_parquet(root / f"data/clean/snli/{split}.parquet")
        prediction = model.predict(tfidf.transform(frame.hypothesis))
        metrics[f"{split}_accuracy"] = accuracy_score(frame.label, prediction)
        metrics[f"{split}_macro_f1"] = f1_score(
            frame.label, prediction, labels=[0, 1, 2], average="macro", zero_division=0
        )
        output = frame[["example_id", "premise", "hypothesis", "label"]].copy()
        output["prediction"] = prediction
        output.to_parquet(out / f"hypothesis_only_{split}_predictions.parquet", index=False)
    (out / "hypothesis_only_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)


def scores(matrix):
    """Accuracy and three-class Macro-F1 from an integer confusion matrix."""
    diagonal = np.diag(matrix)
    denom = matrix.sum(axis=0) + matrix.sum(axis=1)
    f1 = np.divide(2 * diagonal, denom, out=np.zeros(3), where=denom != 0)
    return np.array([diagonal.sum() / matrix.sum(), f1.mean()])


def holm(values):
    order = np.argsort(values)
    adjusted = np.empty(len(values))
    previous = 0.0
    for rank, index in enumerate(order):
        previous = max(previous, (len(values) - rank) * values[index])
        adjusted[index] = min(1.0, previous)
    return adjusted


def analyse(test, predictions, out, repeats, seed):
    y = test.label.to_numpy(dtype=int)
    names = list(predictions)
    group_ids, group_names = pd.factorize(test.premise.map(normalized), sort=True)
    count = len(group_names)
    grouped = np.zeros((len(names), count, 9), dtype=np.int64)
    point = np.empty((len(names), 2))
    for m, name in enumerate(names):
        np.add.at(grouped[m], (group_ids, 3 * y + predictions[name]), 1)
        point[m] = scores(grouped[m].sum(axis=0).reshape(3, 3))
    rng = np.random.default_rng(seed)
    samples = np.empty((repeats, len(names), 2))
    for b in range(repeats):
        # Shared resample across models: paired inference; all rows in each group retained.
        weights = np.bincount(rng.integers(0, count, count), minlength=count)
        for m in range(len(names)):
            samples[b, m] = scores((weights @ grouped[m]).reshape(3, 3))
    rows = []
    for m, name in enumerate(names):
        row = {"model": name, "test_pairs": len(test)}
        for k, metric in enumerate(["accuracy", "macro_f1"]):
            low, high = np.quantile(samples[:, m, k], [0.025, 0.975])
            row.update({metric: point[m, k], f"{metric}_ci_low": low, f"{metric}_ci_high": high})
        rows.append(row)
    pd.DataFrame(rows).to_csv(out / "bootstrap_metrics.csv", index=False)
    differences = []
    tests = []
    for a, b in combinations(range(len(names)), 2):
        row = {"model_a": names[a], "model_b": names[b], "direction": "b minus a"}
        for k, metric in enumerate(["accuracy", "macro_f1"]):
            low, high = np.quantile(samples[:, b, k] - samples[:, a, k], [0.025, 0.975])
            row.update(
                {
                    f"{metric}_difference": point[b, k] - point[a, k],
                    f"{metric}_ci_low": low,
                    f"{metric}_ci_high": high,
                }
            )
        differences.append(row)
        ca, cb = predictions[names[a]] == y, predictions[names[b]] == y
        a_only, b_only = int((ca & ~cb).sum()), int((~ca & cb).sum())
        discordant = a_only + b_only
        p = binomtest(a_only, discordant, p=0.5).pvalue if discordant else 1.0
        tests.append(
            {
                "model_a": names[a],
                "model_b": names[b],
                "a_only_correct": a_only,
                "b_only_correct": b_only,
                "p_value": p,
            }
        )
    # Original three-model comparisons form one family; optional diagnostic comparisons another.
    for original in [True, False]:
        selected = [
            i
            for i, row in enumerate(tests)
            if ("hypothesis_only" not in (row["model_a"], row["model_b"])) == original
        ]
        if selected:
            adjusted = holm([tests[i]["p_value"] for i in selected])
            for i, value in zip(selected, adjusted, strict=True):
                tests[i]["p_holm"] = value
                tests[i]["family"] = "original_models" if original else "diagnostic"
    pd.DataFrame(differences).to_csv(out / "paired_bootstrap_differences.csv", index=False)
    pd.DataFrame(tests).to_csv(out / "mcnemar_exploratory.csv", index=False)
    print("\nCLUSTER BOOTSTRAP (95% percentile intervals)")
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nPAIRED DIFFERENCES")
    print(pd.DataFrame(differences).to_string(index=False))
    print("\nEXPLORATORY MCNEMAR (example independence assumed)")
    print(pd.DataFrame(tests).to_string(index=False))
    meta = {
        "bootstrap_repeats": repeats,
        "seed": seed,
        "groups": count,
        "grouping": "NFC + whitespace + casefold normalized premise",
        "interval": "pointwise 95% paired cluster percentile bootstrap",
        "limitations": [
            "Conditional on these fitted models; not training-seed uncertainty",
            "Premise groups are proxies, not original image groups",
            "Pairwise intervals are not multiplicity-adjusted",
            "McNemar is exploratory: rows may not be independent",
        ],
        "mcnemar": "Exact two-sided binomial test; Holm within each reported family",
    }
    (out / "statistics_protocol.json").write_text(json.dumps(meta, indent=2) + "\n")


def error_slices(test, predictions, out):
    """Exploratory descriptive slices, with sample sizes; no retuning."""
    lengths = test.hypothesis.str.split().str.len()
    negation = test.hypothesis.str.contains(
        r"\b(?:no|not|never|nobody|nothing|neither|nor|without)\b|n['’]t\b",
        case=False,
        regex=True,
    )
    rows = []
    groups = {
        "hypothesis_words_1_5": lengths <= 5,
        "hypothesis_words_6_10": (lengths > 5) & (lengths <= 10),
        "hypothesis_words_11_plus": lengths > 10,
        "negation_marker": negation,
        "no_negation_marker": ~negation,
    }
    for category, mask in groups.items():
        if not mask.any():
            continue
        target = test.loc[mask, "label"].to_numpy(dtype=int)
        for name, prediction in predictions.items():
            rows.append(
                {
                    "slice": category,
                    "model": name,
                    "pairs": int(mask.sum()),
                    "entailment": int((target == 0).sum()),
                    "neutral": int((target == 1).sum()),
                    "contradiction": int((target == 2).sum()),
                    "accuracy": accuracy_score(target, prediction[mask]),
                    "macro_f1": f1_score(
                        target, prediction[mask], labels=[0, 1, 2], average="macro", zero_division=0
                    ),
                }
            )
    pd.DataFrame(rows).to_csv(out / "descriptive_error_slices.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--hypothesis-only", action="store_true")
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.bootstrap_repeats < 100:
        parser.error("Use at least 100 bootstrap replicates (2000 recommended)")
    root = args.root.resolve()
    out = root / "reports/metrics/snli/additional_checks"
    out.mkdir(parents=True, exist_ok=True)
    if args.hypothesis_only:
        with threadpool_limits(limits=4), warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            hypothesis_only(root, out)
    test = load_test(root)
    predictions = {}
    for name in ["baseline", "textcnn", "distilbert"]:
        file = root / f"artifacts/snli/{name}_test_predictions.parquet"
        predictions[name] = align_predictions(test, pd.read_parquet(file))
        expected = json.loads((root / f"reports/metrics/snli/test/{name}_metrics.json").read_text())
        actual = [
            accuracy_score(test.label, predictions[name]),
            f1_score(test.label, predictions[name], average="macro"),
        ]
        if not np.allclose(
            actual, [expected["accuracy"], expected["macro_f1"]], atol=1e-10, rtol=0
        ):
            raise ValueError(f"Recorded metrics mismatch for {name}")
    if args.hypothesis_only:
        predictions["hypothesis_only"] = align_predictions(
            test, pd.read_parquet(out / "hypothesis_only_test_predictions.parquet")
        )
    analyse(test, predictions, out, args.bootstrap_repeats, args.seed)
    error_slices(test, predictions, out)
    print(f"\nSaved diagnostics to {out}")
    print("Original model weights and predictions were not modified.")


if __name__ == "__main__":
    main()
