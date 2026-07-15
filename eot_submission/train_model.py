import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import extract_features, load_wav


def read_training_data(data_root):
    records = []
    cache = {}
    for language_dir in sorted(Path(data_root).iterdir()):
        labels_path = language_dir / "labels.csv"
        if not labels_path.exists():
            continue
        labels = pd.read_csv(labels_path)
        for row in labels.itertuples(index=False):
            audio_path = language_dir / row.audio_file
            if audio_path not in cache:
                cache[audio_path] = load_wav(audio_path)
            x, sr = cache[audio_path]
            records.append({
                "language": language_dir.name.lower(),
                "turn_id": str(row.turn_id),
                "pause_index": int(row.pause_index),
                "x": extract_features(x, sr, row.pause_start, row.pause_index),
                "y": 1 if row.label == "eot" else 0,
                "hold_duration": float(row.pause_end - row.pause_start),
            })
    if not records:
        raise ValueError("No language folders with labels.csv were found")
    return records


def sample_weights(y, durations):
    weights = np.ones(len(y), dtype=np.float64)
    hold = y == 0
    weights[hold] = 1.25
    weights[hold & (durations >= 0.60)] = 2.0
    weights[hold & (durations >= 0.80)] = 3.5
    weights[hold & (durations >= 1.20)] = 5.0
    return weights


def fit_ensemble(X, y, durations, seed):
    weights = sample_weights(y, durations)
    logistic = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(C=0.12, penalty="l1", solver="liblinear", max_iter=1500, random_state=seed),
    )
    trees = ExtraTreesClassifier(
        n_estimators=350,
        min_samples_leaf=5,
        max_features=0.60,
        random_state=seed + 1,
        n_jobs=1,
    )
    boosting = make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingClassifier(
            max_iter=130,
            learning_rate=0.045,
            max_leaf_nodes=7,
            min_samples_leaf=12,
            l2_regularization=3.0,
            random_state=seed + 2,
        ),
    )
    logistic.fit(X, y, logisticregression__sample_weight=weights)
    trees.fit(X, y, sample_weight=weights)
    boosting.fit(X, y, histgradientboostingclassifier__sample_weight=weights)
    return [logistic, trees, boosting]


def _imputer_data(imputer):
    return {"statistics": np.asarray(imputer.statistics_, dtype=np.float64)}


def _export_logistic(pipeline):
    imputer = pipeline.named_steps["simpleimputer"]
    scaler = pipeline.named_steps["standardscaler"]
    classifier = pipeline.named_steps["logisticregression"]
    return {
        "kind": "logistic",
        "imputer": _imputer_data(imputer),
        "mean": np.asarray(scaler.mean_, dtype=np.float64),
        "scale": np.asarray(scaler.scale_, dtype=np.float64),
        "coef": np.asarray(classifier.coef_[0], dtype=np.float64),
        "intercept": float(classifier.intercept_[0]),
    }


def _export_forest(classifier):
    trees = []
    for estimator in classifier.estimators_:
        tree = estimator.tree_
        values = np.asarray(tree.value[:, 0, :], dtype=np.float64)
        totals = values.sum(axis=1)
        p1 = np.divide(values[:, 1], totals, out=np.zeros(len(values)), where=totals > 0)
        trees.append({
            "left": np.asarray(tree.children_left, dtype=np.int32),
            "right": np.asarray(tree.children_right, dtype=np.int32),
            "feature": np.asarray(tree.feature, dtype=np.int32),
            "threshold": np.asarray(tree.threshold, dtype=np.float64),
            "missing_left": np.asarray(tree.missing_go_to_left, dtype=np.uint8),
            "p1": p1,
        })
    return {"kind": "forest", "trees": trees}


def _export_boosting(pipeline):
    imputer = pipeline.named_steps["simpleimputer"]
    classifier = pipeline.named_steps["histgradientboostingclassifier"]
    trees = []
    for stage in classifier._predictors:
        nodes = stage[0].nodes
        trees.append({
            "left": np.asarray(nodes["left"], dtype=np.int32),
            "right": np.asarray(nodes["right"], dtype=np.int32),
            "feature": np.asarray(nodes["feature_idx"], dtype=np.int32),
            "threshold": np.asarray(nodes["num_threshold"], dtype=np.float64),
            "missing_left": np.asarray(nodes["missing_go_to_left"], dtype=np.uint8),
            "is_leaf": np.asarray(nodes["is_leaf"], dtype=np.uint8),
            "value": np.asarray(nodes["value"], dtype=np.float64),
        })
    return {
        "kind": "boosting",
        "imputer": _imputer_data(imputer),
        "baseline": float(classifier._baseline_prediction.ravel()[0]),
        "trees": trees,
    }


def export_ensemble(ensemble):
    return [_export_logistic(ensemble[0]), _export_forest(ensemble[1]), _export_boosting(ensemble[2])]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--out", default="eot_model.pkl")
    args = parser.parse_args()

    records = read_training_data(args.data_root)
    X = np.vstack([row["x"] for row in records])
    y = np.asarray([row["y"] for row in records], dtype=np.int64)
    durations = np.asarray([row["hold_duration"] for row in records], dtype=np.float64)
    languages = np.asarray([row["language"] for row in records])

    generic = fit_ensemble(X, y, durations, 17)
    model = {"version": 2, "feature_count": X.shape[1], "generic": export_ensemble(generic), "languages": {}}
    for language in sorted(set(languages)):
        mask = languages == language
        ensemble = fit_ensemble(X[mask], y[mask], durations[mask], 31 + len(model["languages"]) * 10)
        model["languages"][str(language)] = export_ensemble(ensemble)

    with open(args.out, "wb") as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"trained on {len(y)} pauses with {X.shape[1]} causal features -> {args.out}")


if __name__ == "__main__":
    main()
