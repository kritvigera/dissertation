"""Stage 07 experiments: the ML1 feature judge, evaluated properly.

The headline model is unchanged in kind (gradient boosting on six
scale-invariant descriptors); what changed is everything an examiner
would attack:

  - negatives are generated in GEOMETRY space (transplant, jitter, shear,
    dropout of the column coordinates) before any feature is computed,
    replacing the earlier feature-space corruptions that overlapped three
    of the six features the model then read;
  - the split is grouped by plan, repeated over five seeds, and reported
    as mean +/- std (L11, L12);
  - PR-AUC is reported beside ROC AUC (L13);
  - a leave-one-source-out transfer (train Swiss -> test CubiCasa and the
    reverse) is reported (L14);
  - the trivial single-feature baseline (a depth-1 stump on each feature)
    is reported beside the headline number, always.

Writes data/ml1_results.csv and data/ml1_feature_baselines.csv.

Run:  .venv/bin/python 07_layout_classifier/train_ml1.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.tree import DecisionTreeClassifier

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "09_full_pipeline"))
import pipeline as P                                    # noqa: E402

DATA = HERE.parent / "data"
N_SPLITS = 5


def build_table(seed=42):
    corpus = P.load_judge_corpus()
    rows = P.build_judge_rows(corpus, seed)
    feats, labels, groups, sources, kinds = [], [], [], [], []
    for points, label, group, source, kind in rows:
        f = P.layout_features(points)
        if f is None:
            continue
        feats.append(f)
        labels.append(label)
        groups.append(group)
        sources.append(source)
        kinds.append(kind)
    X = pd.DataFrame(feats)[P.LAYOUT_FEATURES]
    return (X, np.asarray(labels), np.asarray(groups),
            np.asarray(sources), np.asarray(kinds))


def scores(model, X, y):
    p = model.predict_proba(X)[:, 1]
    return {"roc_auc": float(roc_auc_score(y, p)),
            "pr_auc": float(average_precision_score(y, p)),
            "accuracy": float(((p > 0.5) == y).mean())}


def main():
    X, y, groups, sources, kinds = build_table()
    print(f"{len(X)} rows ({y.sum():.0f} positive) over "
          f"{len(np.unique(groups))} plans; negative kinds: "
          f"{dict(pd.Series(kinds[y == 0]).value_counts())}")

    rows = []
    for split_seed in range(N_SPLITS):
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2,
                                random_state=split_seed)
        tr, te = next(gss.split(X, y, groups=groups))
        model = GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                           random_state=42
                                           ).fit(X.iloc[tr], y[tr])
        rows.append({"experiment": "grouped-split", "split": split_seed,
                     "train": "pooled", "test": "pooled",
                     "n_test": len(te), **scores(model, X.iloc[te], y[te])})
        for s in np.unique(sources):
            m = te[sources[te] == s]
            rows.append({"experiment": "grouped-split", "split": split_seed,
                         "train": "pooled", "test": s, "n_test": len(m),
                         **scores(model, X.iloc[m], y[m])})
        # single-feature stump baseline on the same split, best feature
        best = None
        for f in P.LAYOUT_FEATURES:
            stump = DecisionTreeClassifier(max_depth=1, random_state=42
                                           ).fit(X.iloc[tr][[f]], y[tr])
            sc = scores(stump, X.iloc[te][[f]], y[te])
            sc.update({"experiment": "stump-baseline", "split": split_seed,
                       "train": "pooled", "test": f, "n_test": len(te)})
            rows.append(sc)
            if best is None or sc["roc_auc"] > best["roc_auc"]:
                best = dict(sc)
        best.update({"experiment": "stump-best",
                     "test": f"best={best['test']}"})
        rows.append(best)

    # leave-one-source-out (single deterministic run per direction)
    for train_s, test_s in (("Swiss", "CubiCasa"), ("CubiCasa", "Swiss")):
        tr = np.flatnonzero(sources == train_s)
        te = np.flatnonzero(sources == test_s)
        model = GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                           random_state=42
                                           ).fit(X.iloc[tr], y[tr])
        rows.append({"experiment": "leave-one-source-out", "split": 0,
                     "train": train_s, "test": test_s, "n_test": len(te),
                     **scores(model, X.iloc[te], y[te])})

    # per-negative-kind difficulty, pooled model, split 0
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    tr, te = next(gss.split(X, y, groups=groups))
    model = GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                       random_state=42
                                       ).fit(X.iloc[tr], y[tr])
    p = model.predict_proba(X.iloc[te])[:, 1]
    for k in P.NEGATIVE_KINDS:
        mask = (kinds[te] == k) | (y[te] == 1)
        rows.append({"experiment": "per-negative-kind", "split": 0,
                     "train": "pooled", "test": k,
                     "n_test": int((kinds[te] == k).sum()),
                     "roc_auc": float(roc_auc_score(y[te][mask], p[mask])),
                     "pr_auc": float(average_precision_score(
                         y[te][mask], p[mask])),
                     "accuracy": float(((p[mask] > 0.5) ==
                                        y[te][mask]).mean())})

    out = pd.DataFrame(rows)
    out.to_csv(DATA / "ml1_results.csv", index=False)

    pooled = out[(out.experiment == "grouped-split") &
                 (out.test == "pooled")]
    stump = out[out.experiment == "stump-best"]
    print("\nML1 pooled grouped-split over "
          f"{N_SPLITS} seeds: ROC AUC {pooled.roc_auc.mean():.3f} "
          f"+/- {pooled.roc_auc.std():.3f}, "
          f"PR AUC {pooled.pr_auc.mean():.3f} "
          f"+/- {pooled.pr_auc.std():.3f}")
    print(f"best single-feature stump: ROC AUC {stump.roc_auc.mean():.3f} "
          f"+/- {stump.roc_auc.std():.3f} ({stump.test.iloc[0]})")
    print("\nper source / LOSO / per kind:")
    print(out[out.experiment.isin(["leave-one-source-out",
                                   "per-negative-kind"])]
          [["experiment", "train", "test", "roc_auc", "pr_auc"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
