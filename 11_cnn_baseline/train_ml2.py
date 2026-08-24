"""Stage 11 experiments: the ML2 image judge, and the paired
features-versus-pictures comparison.

ML2 trains on the SAME positives and the SAME geometry-space negatives as
ML1 (identical rows, identical labels, built by the shared
build_judge_rows with the same seed), rasterised to a fitted 64x64 canvas:
channel 0 the load-bearing wall fabric, channel 1 the columns as 3x3
blobs. Negatives are corrupted in coordinate space and then rasterised --
never perturbed in pixel space, which would reproduce in pixels the exact
circularity ML1 just had fixed in features.

The features-vs-CNN comparison is PAIRED: the boosting model is fitted on
the features of exactly these rows and scored on exactly the same grouped
split indices, which repairs the two protocol defects of the earlier
stage 11 (negatives resampled from a drifted RNG stream, and a
resolution comparison that changed three things at once).

Writes data/ml2_results.csv and (via pipeline.train_ml2) the judge the
pipeline's V2 variant loads.

Run:  .venv/bin/python 11_cnn_baseline/train_ml2.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "09_full_pipeline"))
import pipeline as P                                    # noqa: E402

DATA = HERE.parent / "data"
N_SPLITS = 3          # CNN fits are minutes, not seconds; 3 seeds, stated


def build_images(seed=42):
    corpus = P.load_judge_corpus()
    walls = pd.read_csv(DATA / "lb_walls.csv.gz")
    wall_map = {(s, str(i)): w for s, i, w in
                zip(walls.source, walls.id, walls.walls)}
    key_of = {g: (s, str(i)) for g, (s, i) in
              enumerate(zip(corpus.source, corpus.id))}
    rows = P.build_judge_rows(corpus, seed)
    X = np.zeros((len(rows), P.IMG_SIZE, P.IMG_SIZE, 2), np.float32)
    y = np.zeros(len(rows), np.float32)
    feats, groups, sources = [], [], []
    keep = []
    for j, (points, label, group, source, kind) in enumerate(rows):
        f = P.layout_features(points)
        if f is None:
            continue                       # keep rows identical to ML1's
        w = wall_map.get(key_of[group])
        walls5 = (np.array([[float(v) for v in seg.split()]
                            for seg in w.split(";")])
                  if isinstance(w, str) and w else None)
        X[j] = P.rasterise(points, walls5)
        y[j] = label
        feats.append(f)
        groups.append(group)
        sources.append(source)
        keep.append(j)
    X = X[keep]
    y = y[keep]
    F = pd.DataFrame(feats)[P.LAYOUT_FEATURES]
    return X, y, F, np.asarray(groups), np.asarray(sources)


def fit_cnn(X_tr, y_tr, epochs=25):
    import tensorflow as tf
    model = P.build_ml2_model()
    model.fit(X_tr, y_tr, epochs=epochs, batch_size=64, verbose=0,
              validation_split=0.1,
              callbacks=[tf.keras.callbacks.EarlyStopping(
                  patience=4, restore_best_weights=True)])
    return model


def metric_row(y_true, p):
    return {"roc_auc": float(roc_auc_score(y_true, p)),
            "pr_auc": float(average_precision_score(y_true, p)),
            "accuracy": float(((p > 0.5) == y_true).mean())}


def main():
    t0 = time.time()
    X, y, F, groups, sources = build_images()
    print(f"{len(X)} images of {P.IMG_SIZE}x{P.IMG_SIZE}x2 "
          f"({int(y.sum())} positive) in {time.time() - t0:.0f}s")

    rows = []
    for split_seed in range(N_SPLITS):
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2,
                                random_state=split_seed)
        tr, te = next(gss.split(X, y, groups=groups))
        t1 = time.time()
        cnn = fit_cnn(X[tr], y[tr])
        p_cnn = cnn.predict(X[te], verbose=0).ravel()
        rows.append({"model": "ML2 CNN on pictures", "split": split_seed,
                     "train": "pooled", "test": "pooled", "n_test": len(te),
                     "seconds": time.time() - t1, **metric_row(y[te], p_cnn)})
        for s in np.unique(sources):
            m = sources[te] == s
            rows.append({"model": "ML2 CNN on pictures",
                         "split": split_seed, "train": "pooled", "test": s,
                         "n_test": int(m.sum()),
                         **metric_row(y[te][m], p_cnn[m])})
        # paired control: boosting on the features of the SAME rows/split
        t1 = time.time()
        gbm = GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                         random_state=42
                                         ).fit(F.iloc[tr], y[tr])
        p_gbm = gbm.predict_proba(F.iloc[te])[:, 1]
        rows.append({"model": "ML1 boosting on features (paired)",
                     "split": split_seed, "train": "pooled",
                     "test": "pooled", "n_test": len(te),
                     "seconds": time.time() - t1,
                     **metric_row(y[te], p_gbm)})
        print(f"split {split_seed}: CNN {rows[-len(np.unique(sources)) - 2]['roc_auc']:.3f} "
              f"vs features {rows[-1]['roc_auc']:.3f} ROC AUC")

    # leave-one-source-out
    for train_s, test_s in (("Swiss", "CubiCasa"), ("CubiCasa", "Swiss")):
        tr = np.flatnonzero(sources == train_s)
        te = np.flatnonzero(sources == test_s)
        cnn = fit_cnn(X[tr], y[tr])
        p = cnn.predict(X[te], verbose=0).ravel()
        rows.append({"model": "ML2 CNN on pictures", "split": 0,
                     "train": train_s, "test": test_s, "n_test": len(te),
                     **metric_row(y[te], p)})
        gbm = GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                         random_state=42
                                         ).fit(F.iloc[tr], y[tr])
        pg = gbm.predict_proba(F.iloc[te])[:, 1]
        rows.append({"model": "ML1 boosting on features (paired)",
                     "split": 0, "train": train_s, "test": test_s,
                     "n_test": len(te), **metric_row(y[te], pg)})

    out = pd.DataFrame(rows)
    out.to_csv(DATA / "ml2_results.csv", index=False)
    pooled = out[(out.test == "pooled")]
    print("\npooled over splits:")
    print(pooled.groupby("model")[["roc_auc", "pr_auc", "accuracy"]]
          .agg(["mean", "std"]).round(3).to_string())
    print("\nleave-one-source-out:")
    print(out[out.train.isin(["Swiss", "CubiCasa"])]
          [["model", "train", "test", "roc_auc", "pr_auc"]]
          .round(3).to_string(index=False))

    # finally, train the deployable judge on the full corpus for V2
    print("\ntraining the deployable V2 judge on the full corpus ...")
    P.train_ml2()
    print(f"saved {P.ML2_WEIGHTS}")


if __name__ == "__main__":
    main()
