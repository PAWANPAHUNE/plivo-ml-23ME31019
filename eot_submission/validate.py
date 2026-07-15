import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from train_model import fit_ensemble, read_training_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--out_dir", default="validation")
    args = parser.parse_args()

    records = read_training_data(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for language in sorted({row["language"] for row in records}):
        subset = [row for row in records if row["language"] == language]
        X = np.vstack([row["x"] for row in subset])
        y = np.asarray([row["y"] for row in subset], dtype=np.int64)
        durations = np.asarray([row["hold_duration"] for row in subset], dtype=np.float64)
        groups = np.asarray([row["turn_id"] for row in subset])
        probabilities = np.zeros(len(subset), dtype=np.float64)
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=23)
        for fold, (train_index, test_index) in enumerate(splitter.split(X, y, groups)):
            ensemble = fit_ensemble(
                X[train_index], y[train_index], durations[train_index], 100 + fold * 10
            )
            probabilities[test_index] = np.mean(
                [model.predict_proba(X[test_index])[:, 1] for model in ensemble], axis=0
            )
        path = out_dir / f"predictions_{language}_oof.csv"
        with open(path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["turn_id", "pause_index", "p_eot"])
            for row, probability in zip(subset, probabilities):
                writer.writerow([row["turn_id"], row["pause_index"], f"{probability:.6f}"])
        print(path)


if __name__ == "__main__":
    main()
