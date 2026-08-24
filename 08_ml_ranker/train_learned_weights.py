"""Learn the score-function weights from the corpus, by ranking.

The only non-circular way to learn the weights is learning to rank: for
each real plan, the derived ground-truth arrangement is the positive, and
alternative arrangements enumerated over that same plan's grid are the
negatives. A pairwise logistic model is fitted on the differences of the
robustly normalised score terms, so that penalty(ground truth) <
penalty(alternative) as often as the data can make it. The model is linear
and intercept-free on term differences, which keeps the learned weights
readable and directly comparable against the algorithmic vector of ones --
that comparison is itself a thesis result (stage 12 reports both under the
same protocol).

What this is NOT trained on, deliberately:
  - the algorithmic score's own output (a regression onto it would be a
    tautology);
  - the real-versus-corrupted signal the two judges use (stages 07 and 11).
    The judges' negatives are transplants and coordinate perturbations;
    the ranker's negatives are line subsets and placement policies over
    the same plan. The two signals overlap in one place -- both contain
    full-lattice-like arrangements -- and the limitations section says so.

Writes:
  data/learned_weights.json        the weights score_mode="learned" loads
  data/rank_pairs_summary.csv      pairwise accuracy: split x source x kind

Run:  .venv/bin/python 08_ml_ranker/train_learned_weights.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "09_full_pipeline"))
import pipeline as P                                    # noqa: E402

DATA = HERE.parent / "data"
SEED = 42
DROP_PROBS = (0.2, 0.4)      # interior-line dropout rates for alternatives


def parse_walls5(text):
    if not isinstance(text, str) or not text.strip():
        return np.zeros((0, 5))
    return np.array([[float(v) for v in seg.split()]
                     for seg in text.split(";")])


def surrogate_grid(points, walls5):
    """A minimal grid dict for score_terms, built from the plan's own
    ground-truth lines and load-bearing walls (already in one frame)."""
    xs, ys = np.unique(points[:, 0]), np.unique(points[:, 1])
    ends = np.vstack([walls5[:, 0:2], walls5[:, 2:4]])
    span = max(ends[:, 0].max() - ends[:, 0].min(),
               ends[:, 1].max() - ends[:, 1].min())
    nodes = np.array([(x, y) for x in xs for y in ys])
    in_wall = P.nodes_in_walls(nodes, walls5, slack=0.015 * span)
    return {"x_lines": xs, "y_lines": ys, "nodes": nodes,
            "in_wall": in_wall, "blocked": np.zeros(len(nodes), bool),
            "footprint": (float(ends[:, 0].max() - ends[:, 0].min()),
                          float(ends[:, 1].max() - ends[:, 1].min())),
            "centre": (float((ends[:, 0].max() + ends[:, 0].min()) / 2),
                       float((ends[:, 1].max() + ends[:, 1].min()) / 2))}


def alternatives(points, grid, rng, n_stories=1):
    """Enumerated alternatives over the same plan's grid: the full lattice,
    plus line subsets under both placement policies."""
    xs, ys = grid["x_lines"], grid["y_lines"]
    nodes, in_wall = grid["nodes"], grid["in_wall"]
    gt_key = {tuple(r) for r in np.round(points, 2).tolist()}
    out = []

    def add(cols):
        cols = np.asarray(cols, float)
        if len(cols) < 4:
            return
        if len(np.unique(cols[:, 0])) < 2 or len(np.unique(cols[:, 1])) < 2:
            return
        if {tuple(r) for r in np.round(cols, 2).tolist()} == gt_key:
            return                                      # identical to GT
        out.append({"x_lines": np.unique(cols[:, 0]),
                    "y_lines": np.unique(cols[:, 1]),
                    "columns": cols, "n_stories": n_stories})

    add(nodes)                                          # the full lattice
    for p in DROP_PROBS:
        keep_x = np.r_[True, rng.random(len(xs) - 2) > p, True]
        keep_y = np.r_[True, rng.random(len(ys) - 2) > p, True]
        sub_x, sub_y = set(xs[keep_x]), set(ys[keep_y])
        mask = np.array([(x in sub_x) and (y in sub_y) for x, y in nodes])
        add(nodes[mask])                                # lattice placement
        add(nodes[mask & in_wall])                      # walls placement
    return out


def build_pairs(coords, walls, seed=SEED, log=print):
    rng = np.random.default_rng(seed)
    wall_map = {(s, str(i)): w for s, i, w in
                zip(walls.source, walls.id, walls.walls)}
    X, groups, sources, kinds = [], [], [], []
    skipped = 0
    for g, row in enumerate(coords.itertuples(index=False)):
        w = wall_map.get((row.source, str(row.id)))
        walls5 = parse_walls5(w)
        pts = row.points
        if len(walls5) < 2 or len(np.unique(pts[:, 0])) < 3 \
                or len(np.unique(pts[:, 1])) < 3:
            skipped += 1
            continue
        grid = surrogate_grid(pts, walls5)
        gt = {"x_lines": grid["x_lines"], "y_lines": grid["y_lines"],
              "columns": pts, "n_stories": 1}
        alts = alternatives(pts, grid, rng)
        if not alts:
            skipped += 1
            continue
        table = P.score_arrangements([gt] + alts, grid,
                                     weights=P.ALGO_WEIGHTS)
        z = table[[f"z_{t}" for t in P.SCORE_TERMS]].values
        for j, alt in enumerate(alts, start=1):
            X.append(z[j] - z[0])                       # alt minus GT
            groups.append(g)
            sources.append(row.source)
            kinds.append("lattice-full" if j == 1 else
                         ("walls-subset" if _is_wall_subset(alt, grid)
                          else "lattice-subset"))
        if g % 2000 == 0 and g:
            log(f"  {g} plans, {len(X)} pairs")
    log(f"pairs built: {len(X)} from {len(coords) - skipped} plans "
        f"({skipped} skipped)")
    return (np.asarray(X), np.asarray(groups),
            np.asarray(sources), np.asarray(kinds))


def _is_wall_subset(alt, grid):
    key = {tuple(r) for r in np.round(grid["nodes"][grid["in_wall"]],
                                      6).tolist()}
    return all(tuple(r) in key
               for r in np.round(alt["columns"], 6).tolist())


def fit_weights(X, y_dummy=None):
    """Pairwise logistic on symmetrised differences, no intercept."""
    Xs = np.vstack([X, -X])
    ys = np.r_[np.ones(len(X)), np.zeros(len(X))]
    model = LogisticRegression(fit_intercept=False, max_iter=2000)
    model.fit(Xs, ys)
    return model.coef_.ravel()


def main():
    coords = P.load_judge_corpus()                      # drops the 7 demos
    walls = pd.read_csv(DATA / "lb_walls.csv.gz")
    X, groups, sources, kinds = build_pairs(coords, walls)

    rows = []
    coefs = []
    for split_seed in range(5):
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2,
                                random_state=split_seed)
        tr, te = next(gss.split(X, groups=groups))
        w = fit_weights(X[tr])
        coefs.append(w)
        acc = lambda idx: float(np.mean(X[idx] @ w > 0))
        rows.append({"split": split_seed, "scope": "train", "kind": "all",
                     "pairs": len(tr), "gt_ranked_better": acc(tr)})
        rows.append({"split": split_seed, "scope": "test", "kind": "all",
                     "pairs": len(te), "gt_ranked_better": acc(te)})
        for s in np.unique(sources):
            m = te[sources[te] == s]
            rows.append({"split": split_seed, "scope": "test", "kind": s,
                         "pairs": len(m), "gt_ranked_better": acc(m)})
        for k in np.unique(kinds):
            m = te[kinds[te] == k]
            rows.append({"split": split_seed, "scope": "test", "kind": k,
                         "pairs": len(m), "gt_ranked_better": acc(m)})

    coefs = np.asarray(coefs)
    mean_w = coefs.mean(axis=0)

    # The deployed vector is fitted on the canonical 80% training partition
    # (data/train_test_split.csv), so that the weights the pipeline uses
    # never saw the 20% test partition or the demo plans. The five repeated
    # splits above remain the reported cross-validated accuracy.
    canon = P.load_judge_corpus(partition="train")
    keep = set(zip(canon.source.astype(str), canon.id.astype(str)))
    plan_key = [(s_, str(i_)) for s_, i_ in zip(coords.source, coords.id)]
    keyed = {}
    for g, k in enumerate(plan_key):
        keyed[g] = k
    in_train = np.array([keyed[g] in keep for g in groups])
    w_dep = fit_weights(X[in_train])
    # normalise to mean absolute weight 1, the same scale as the
    # algorithmic vector of ones; ranking is invariant to the scale
    norm_w = w_dep / np.mean(np.abs(w_dep))
    weights = {t: float(w) for t, w in zip(P.SCORE_TERMS, norm_w)}

    # the algorithmic vector's pairwise accuracy, same test folds
    algo_vec = np.ones(len(P.SCORE_TERMS))
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    _, te = next(gss.split(X, groups=groups))
    algo_acc = float(np.mean(X[te] @ algo_vec > 0))

    summary = pd.DataFrame(rows)
    summary.to_csv(DATA / "rank_pairs_summary.csv", index=False)
    out = {
        "weights": weights,
        "raw_mean_coefficients": {t: float(w) for t, w in
                                  zip(P.SCORE_TERMS, mean_w)},
        "coefficient_std_over_splits": {t: float(s) for t, s in
                                        zip(P.SCORE_TERMS,
                                            coefs.std(axis=0))},
        "pairwise_accuracy_test_mean": float(
            summary[(summary.scope == "test") &
                    (summary.kind == "all")].gt_ranked_better.mean()),
        "pairwise_accuracy_test_std": float(
            summary[(summary.scope == "test") &
                    (summary.kind == "all")].gt_ranked_better.std()),
        "algorithmic_vector_accuracy_same_fold": algo_acc,
        "n_pairs": int(len(X)),
        "n_plans": int(len(np.unique(groups))),
        "seed": SEED,
        "note": ("Pairwise logistic regression on differences of robustly "
                 "normalised score terms; positives are derived ground "
                 "truths, negatives are enumerated line-subset and "
                 "placement alternatives over the same plan's grid. "
                 "Never trained on the algorithmic score's output. The "
                 "deployed vector is fitted on the canonical 80% training "
                 "partition; the accuracy figures are the mean over five "
                 "repeated grouped splits."),
    }
    (DATA / "learned_weights.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
