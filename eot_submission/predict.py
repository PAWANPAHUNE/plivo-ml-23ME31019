import argparse
import csv
import pickle
from pathlib import Path

import numpy as np

from features import extract_features, load_wav


def _sigmoid(values):
    values = np.clip(values, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-values))


def _impute(X, data):
    X = np.asarray(X, dtype=np.float64).copy()
    mask = ~np.isfinite(X)
    if mask.any():
        X[mask] = np.take(data["statistics"], np.where(mask)[1])
    return X


def _tree_values(tree, X, leaf_key):
    output = np.empty(len(X), dtype=np.float64)
    for row_index, row in enumerate(X):
        node = 0
        while True:
            if "is_leaf" in tree:
                leaf = bool(tree["is_leaf"][node])
            else:
                leaf = tree["left"][node] < 0 or tree["feature"][node] < 0
            if leaf:
                output[row_index] = tree[leaf_key][node]
                break
            feature = int(tree["feature"][node])
            value = row[feature]
            go_left = bool(tree["missing_left"][node]) if not np.isfinite(value) else value <= tree["threshold"][node]
            node = int(tree["left"][node] if go_left else tree["right"][node])
    return output


def _predict_estimator(estimator, X):
    kind = estimator["kind"]
    if kind == "logistic":
        values = _impute(X, estimator["imputer"])
        values = (values - estimator["mean"]) / estimator["scale"]
        return _sigmoid(values @ estimator["coef"] + estimator["intercept"])
    if kind == "forest":
        return np.mean([_tree_values(tree, X, "p1") for tree in estimator["trees"]], axis=0)
    if kind == "boosting":
        values = _impute(X, estimator["imputer"])
        raw = np.full(len(values), estimator["baseline"], dtype=np.float64)
        for tree in estimator["trees"]:
            raw += _tree_values(tree, values, "value")
        return _sigmoid(raw)
    raise ValueError(f"Unknown estimator kind: {kind}")


def choose_ensemble(model, data_dir, turn_ids):
    text = (Path(data_dir).name + " " + " ".join(turn_ids[:20])).lower()
    if "hindi" in text or any(turn_id.lower().startswith("hi__") for turn_id in turn_ids):
        return model["languages"].get("hindi", model["generic"])
    if "english" in text or any(turn_id.lower().startswith("en__") for turn_id in turn_ids):
        return model["languages"].get("english", model["generic"])
    return model["generic"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out", default="predictions.csv")
    parser.add_argument("--model", default=str(Path(__file__).with_name("eot_model.pkl")))
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    with open(data_dir / "labels.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"turn_id", "audio_file", "pause_index", "pause_start"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"labels.csv must contain {sorted(required)}")

    with open(args.model, "rb") as handle:
        model = pickle.load(handle)
    if model.get("version") != 2:
        raise ValueError("Expected portable model version 2")
    ensemble = choose_ensemble(model, str(data_dir), [row["turn_id"] for row in rows])

    cache = {}
    feature_rows = []
    for row in rows:
        audio_path = data_dir / row["audio_file"]
        if audio_path not in cache:
            cache[audio_path] = load_wav(audio_path)
        x, sr = cache[audio_path]
        feature_rows.append(extract_features(x, sr, float(row["pause_start"]), int(row["pause_index"])))

    X = np.vstack(feature_rows)
    if X.shape[1] != model["feature_count"]:
        raise ValueError("Feature version does not match the saved model")
    probabilities = np.mean([_predict_estimator(estimator, X) for estimator in ensemble], axis=0)
    probabilities = np.clip(probabilities, 0.001, 0.999)

    with open(args.out, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["turn_id", "pause_index", "p_eot"])
        for row, probability in zip(rows, probabilities):
            writer.writerow([row["turn_id"], row["pause_index"], f"{probability:.6f}"])
    print(f"wrote {len(rows)} predictions -> {args.out}")


if __name__ == "__main__":
    main()
