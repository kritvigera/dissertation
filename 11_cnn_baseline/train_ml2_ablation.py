"""The wall-channel ablation: the same CNN, the same rows, the same
splits, with channel 0 (the wall fabric) blanked.

The two-channel image judge sees something the six scale-invariant
features deliberately cannot: whether the columns register with the walls.
If its advantage over the feature judge comes mostly from that channel,
the honest statement is "pictures win because they carry the plan context,
not because convolution reads column patterns better" -- and this ablation
is how we find out. Appends rows to data/ml2_results.csv with the model
name "ML2 CNN, columns only".

Run:  .venv/bin/python 11_cnn_baseline/train_ml2_ablation.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "09_full_pipeline"))
sys.path.insert(0, str(HERE))
import pipeline as P                                    # noqa: E402
from train_ml2 import build_images, fit_cnn, metric_row, N_SPLITS  # noqa: E402

DATA = HERE.parent / "data"


def main():
    X, y, F, groups, sources = build_images()
    X = X.copy()
    X[..., 0] = 0.0                       # blank the wall channel
    rows = []
    for split_seed in range(N_SPLITS):
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2,
                                random_state=split_seed)
        tr, te = next(gss.split(X, y, groups=groups))
        cnn = fit_cnn(X[tr], y[tr])
        p = cnn.predict(X[te], verbose=0).ravel()
        rows.append({"model": "ML2 CNN, columns only", "split": split_seed,
                     "train": "pooled", "test": "pooled", "n_test": len(te),
                     **metric_row(y[te], p)})
        print(f"split {split_seed}: columns-only CNN "
              f"ROC AUC {rows[-1]['roc_auc']:.3f}")
    for train_s, test_s in (("Swiss", "CubiCasa"), ("CubiCasa", "Swiss")):
        tr = np.flatnonzero(sources == train_s)
        te = np.flatnonzero(sources == test_s)
        cnn = fit_cnn(X[tr], y[tr])
        p = cnn.predict(X[te], verbose=0).ravel()
        rows.append({"model": "ML2 CNN, columns only", "split": 0,
                     "train": train_s, "test": test_s, "n_test": len(te),
                     **metric_row(y[te], p)})

    out = pd.concat([pd.read_csv(DATA / "ml2_results.csv"),
                     pd.DataFrame(rows)], ignore_index=True)
    out.to_csv(DATA / "ml2_results.csv", index=False)
    pooled = out[out.test == "pooled"]
    print(pooled.groupby("model")[["roc_auc", "pr_auc"]]
          .agg(["mean", "std"]).round(3).to_string())


if __name__ == "__main__":
    main()
