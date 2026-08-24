"""Stage 13: the evaluation protocol. Everything else reports to this.

Four pipeline variants share every upstream stage and differ only in the
judge that re-orders the score function's top 15:

    V1  score -> ML1 feature judge -> best
    V2  score -> ML2 image judge   -> best
    V3  score only                 -> best   (the baseline to beat)
    V4  score -> quantum judge     -> best

Each variant runs end to end on the seven held-out demo plans, under both
score modes (algorithmic and learned weights), and is measured on two
independent axes:

  ACCURACY against the derived ground-truth arrangement
    - reachability first: is the ground truth inside the enumerated
      candidate set at all? Every accuracy number is capped by this rate,
      so it is reported before any of them.
    - top-1 / top-5 / rank of the ground truth in the variant's full
      ordering, on the pool with the ground truth guaranteed present
      (injected and flagged when not organically reachable).
    - soft geometric agreement: Chamfer distance (metres) and a
      tolerance-matched IoU between the chosen columns and the ground
      truth's, because exact-set recovery is brutal and near misses matter.

Variants are compared with PAIRED tests across plans (Wilcoxon signed-rank
plus an exact sign test); with n = 7 plans the power is limited and the
write-up says so.

Also produced here: the lambda ablation, the +/-50% and leave-one-term-out
weight sensitivity, the 140/160/180/200 mm ground-truth threshold
sensitivity, and the score-vs-judge ranking overlap (loophole L7).

Run:  .venv/bin/python 12_evaluation/evaluate.py            # V1+V3
      .venv/bin/python 12_evaluation/evaluate.py --variants V1,V2,V3,V4
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "09_full_pipeline"))
sys.path.insert(0, str(HERE.parent / "02_data_pipeline"))
import pipeline as P                                    # noqa: E402
import regenerate_corpus as rc                          # noqa: E402

DATA = HERE.parent / "data"
DEMOS = [f"demo_{i:02d}" for i in range(1, 8)]
MATCH_TOL = 0.05             # metres: two columns count as the same place
LAMBDAS = (0.0, 0.1, 0.3, 0.5, 1.0)
THRESHOLDS = (0.140, 0.160, 0.180, 0.200)

# The evaluation span band is deliberately wide (these are dwellings whose
# load-bearing walls can sit 0.13 m apart, not offices on a 6-12 m rhythm)
# and deliberately fixed across plans: deriving it per plan from the ground
# truth would leak the answer into the question, and setting the floor any
# higher would exclude the ground truth's own line set a priori.
EVAL_MIN_SPAN = 0.1
EVAL_MAX_SPAN = 10.0


def eval_constraints(plan):
    return {"n_stories": plan["n_stories"],
            "min_span": EVAL_MIN_SPAN, "max_span": EVAL_MAX_SPAN,
            "unit_scale": 1.0,
            "merge_tol": "auto",                   # the corpus tolerance
            "min_wall_frac": 0.05,                 # the corpus length filter
            "fixed_columns": plan.get("annotated_columns", []),
            "offer_skew_columns": False,           # the ground truth cannot
                                                   # express skew crossings
            "column_free_zones": plan.get("column_free_zones", [])}


def ground_truth_for(plan, threshold=0.160):
    """The derived ground-truth arrangement, re-derived from the demo
    file's own walls with the stage 02 code, so grid, columns and frame
    are consistent with the plan geometry by construction."""
    lb = rc.load_bearing([list(w) for w in plan["walls"]],
                         "absolute", threshold)
    derived = rc.derive_arrangement(lb, plan.get("annotated_columns") or [])
    if derived is None:
        return None
    x_lines, y_lines, _, cols, _ = derived
    return {"x_lines": np.asarray(x_lines, float),
            "y_lines": np.asarray(y_lines, float),
            "columns": np.asarray(cols, float),
            "skew_columns": np.zeros((0, 2)),
            "n_stories": plan["n_stories"],
            "placement": "ground-truth"}


def all_columns(arr):
    c = np.asarray(arr["columns"], float)
    s = np.asarray(arr.get("skew_columns", []), float).reshape(-1, 2)
    return np.vstack([c, s]) if len(s) else c


def greedy_matches(a, b, tol=MATCH_TOL):
    """How many of a's points have their own partner in b within tol."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) == 0 or len(b) == 0:
        return 0
    used = np.zeros(len(b), bool)
    m = 0
    for p in a:
        d = np.hypot(b[:, 0] - p[0], b[:, 1] - p[1])
        d[used] = np.inf
        k = int(d.argmin())
        if d[k] <= tol:
            used[k] = True
            m += 1
    return m


def same_arrangement(a, b, tol=MATCH_TOL):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return len(a) == len(b) and greedy_matches(a, b, tol) == len(a)


def chamfer(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = np.hypot(a[:, None, 0] - b[None, :, 0],
                 a[:, None, 1] - b[None, :, 1])
    return float((d.min(axis=1).mean() + d.min(axis=0).mean()) / 2)


def iou(a, b, tol=MATCH_TOL):
    m = greedy_matches(a, b, tol)
    return m / (len(a) + len(b) - m)


def penalties_for(table, weights):
    z = table[[f"z_{t}" for t in P.SCORE_TERMS]].values
    w = np.array([weights[t] for t in P.SCORE_TERMS])
    return z @ w


def order_for(pool, grid, penalty, judge, lam=0.3, keep=15):
    """The penalty is normalised over the FULL candidate set, so within the
    top 15 its differences are marginal and the judge has real authority;
    normalising within the shortlist would spread near-ties across a full
    IQR and mute the judge at any sensible lambda."""
    base = np.argsort(penalty, kind="stable")
    if judge is None:
        return base
    top = base[:keep]
    p_pl = np.asarray(judge.plausibility([pool[i] for i in top], grid))
    combined = P.robust_z(penalty)[top] + lam * (1 - p_pl)
    return np.concatenate([top[np.argsort(combined, kind="stable")],
                           base[keep:]])


def load_plan(demo):
    return json.loads((DATA / "demo_plans" / f"{demo}.json").read_text())


def prepare_pool(plan, threshold=0.160, max_arrangements=4000):
    """Parse, enumerate, derive the ground truth, and make sure the pool
    contains it (recording whether it got there on its own)."""
    cons = eval_constraints(plan)
    grid = P.parse_plan(plan["walls"], cons)
    cands = P.generate(grid, max_arrangements=max_arrangements)
    gt = ground_truth_for(plan, threshold)
    if gt is None:
        return None
    gt_cols = all_columns(gt)
    gt_idx, reachable = None, False
    for i, c in enumerate(cands):
        if same_arrangement(all_columns(c), gt_cols):
            gt_idx, reachable = i, True
            break
    pool = list(cands)
    if gt_idx is None:
        pool.append(gt)
        gt_idx = len(pool) - 1
    return {"grid": grid, "pool": pool, "gt_idx": gt_idx,
            "reachable": reachable, "gt": gt}


def evaluate_variants(prep, judges, modes, lam=0.3):
    grid, pool, gt_idx = prep["grid"], prep["pool"], prep["gt_idx"]
    table = P.score_arrangements(pool, grid, weights=P.ALGO_WEIGHTS)
    gt_cols = all_columns(pool[gt_idx])
    acc_rows = []
    for mode, weights in modes.items():
        penalty = penalties_for(table, weights)
        for variant, judge in judges.items():
            order = order_for(pool, grid, penalty, judge, lam)
            rank = int(np.where(order == gt_idx)[0][0]) + 1
            best = pool[order[0]]
            best_cols = all_columns(best)
            acc_rows.append({
                "mode": mode, "variant": variant,
                "n_candidates": len(pool), "reachable": prep["reachable"],
                "rank_gt": rank, "top1": rank == 1, "top5": rank <= 5,
                "chamfer_m": chamfer(best_cols, gt_cols),
                "iou": iou(best_cols, gt_cols),
                "best_n_columns": len(best_cols),
                "gt_n_columns": len(gt_cols),
                "best_placement": best.get("placement")})
    return acc_rows, table


def paired_tests(df, metric, variants, modes):
    out = []
    for mode in modes:
        for a in variants:
            for b in variants:
                if a >= b:
                    continue
                xa = df[(df["mode"] == mode) & (df.variant == a)
                        ].sort_values("plan")[metric].values.astype(float)
                xb = df[(df["mode"] == mode) & (df.variant == b)
                        ].sort_values("plan")[metric].values.astype(float)
                diff = xa - xb
                nz = diff[diff != 0]
                if len(nz) == 0:
                    w_p = 1.0
                else:
                    try:
                        w_p = float(stats.wilcoxon(xa, xb).pvalue)
                    except ValueError:
                        w_p = 1.0
                wins = int((diff < 0).sum())
                sign_p = float(stats.binomtest(
                    wins, max(len(nz), 1), 0.5).pvalue) if len(nz) else 1.0
                out.append({"mode": mode, "metric": metric,
                            "pair": f"{a} vs {b}",
                            "mean_a": float(np.mean(xa)),
                            "mean_b": float(np.mean(xb)),
                            "wilcoxon_p": w_p, "sign_p": sign_p,
                            "n_plans": len(xa)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="V1,V3")
    ap.add_argument("--lam", type=float, default=0.3)
    ap.add_argument("--skip-sweeps", action="store_true")
    ap.add_argument("--max-arrangements", type=int, default=4000)
    args = ap.parse_args()
    wanted = args.variants.split(",")

    print("preparing judges ...")
    judges = {}
    for v in wanted:
        if v == "V1":
            judges["V1"] = P.ML1Judge()
        elif v == "V2":
            judges["V2"] = P.ML2Judge()
        elif v == "V3":
            judges["V3"] = None
        elif v == "V4":
            judges["V4"] = P.QuantumJudge()
    modes = {"algo": P.ALGO_WEIGHTS}
    try:
        modes["learned"] = P.load_learned_weights()
    except FileNotFoundError:
        print("  (learned weights not found; algo mode only)")

    acc_all, preps = [], {}
    for demo in DEMOS:
        plan = load_plan(demo)
        prep = prepare_pool(plan, max_arrangements=args.max_arrangements)
        if prep is None:
            print(f"{demo}: no ground truth derivable, skipped")
            continue
        preps[demo] = prep
        a, _ = evaluate_variants(prep, judges, modes, args.lam)
        for r in a:
            r["plan"] = demo
        acc_all += a
        print(f"{demo}: {len(prep['pool'])} candidates, "
              f"reachable={prep['reachable']}, "
              f"gt has {len(all_columns(prep['gt']))} columns")

    acc = pd.DataFrame(acc_all)
    acc.to_csv(DATA / "evaluation_accuracy.csv", index=False)

    reach_rate = np.mean([p["reachable"] for p in preps.values()])
    print(f"\nREACHABILITY: {reach_rate:.0%} of plans "
          f"({int(reach_rate * len(preps))}/{len(preps)})")
    print("\naccuracy (mean over plans):")
    print(acc.groupby(["mode", "variant"])[
        ["rank_gt", "top1", "top5", "chamfer_m", "iou"]]
        .mean().round(3).to_string())

    tests = []
    for metric in ("rank_gt", "chamfer_m", "iou"):
        tests += paired_tests(acc, metric, wanted, modes.keys())
    pd.DataFrame(tests).to_csv(DATA / "evaluation_tests.csv", index=False)

    if args.skip_sweeps:
        return

    # ------------- lambda ablation (V1), under both score modes
    if "V1" in judges:
        rows = []
        for demo, prep in preps.items():
            table = P.score_arrangements(prep["pool"], prep["grid"],
                                         weights=P.ALGO_WEIGHTS)
            for mode, weights in modes.items():
                penalty = penalties_for(table, weights)
                for lam in LAMBDAS:
                    order = order_for(prep["pool"], prep["grid"], penalty,
                                      judges["V1"], lam)
                    rank = int(np.where(order == prep["gt_idx"])[0][0]) + 1
                    rows.append({"plan": demo, "mode": mode, "lambda": lam,
                                 "rank_gt": rank, "top1": rank == 1,
                                 "top5": rank <= 5})
        lam_df = pd.DataFrame(rows)
        lam_df.to_csv(DATA / "evaluation_lambda.csv", index=False)
        print("\nlambda ablation (V1, mean over plans):")
        print(lam_df.groupby(["mode", "lambda"])[
            ["rank_gt", "top1", "top5"]].mean().round(3).to_string())

    # ------------- weight sensitivity (V3, algorithmic weights)
    rows = []
    perturbations = [("nominal", P.ALGO_WEIGHTS)]
    for t in P.SCORE_TERMS:
        for f, tag in ((1.5, "+50%"), (0.5, "-50%"), (0.0, "dropped")):
            w = dict(P.ALGO_WEIGHTS)
            w[t] = w[t] * f
            perturbations.append((f"{t} {tag}", w))
    for demo, prep in preps.items():
        table = P.score_arrangements(prep["pool"], prep["grid"],
                                     weights=P.ALGO_WEIGHTS)
        nominal = penalties_for(table, P.ALGO_WEIGHTS)
        nom_order = np.argsort(nominal, kind="stable")
        for name, w in perturbations:
            pen = penalties_for(table, w)
            order = np.argsort(pen, kind="stable")
            tau = stats.kendalltau(nominal, pen).statistic
            rows.append({"plan": demo, "perturbation": name,
                         "kendall_tau": float(tau),
                         "top1_same": bool(order[0] == nom_order[0]),
                         "top5_overlap": len(set(order[:5])
                                             & set(nom_order[:5])) / 5,
                         "rank_gt": int(np.where(order == prep["gt_idx"]
                                                 )[0][0]) + 1})
    sens = pd.DataFrame(rows)
    sens.to_csv(DATA / "evaluation_weights_sensitivity.csv", index=False)
    print("\nweight sensitivity (V3, mean over plans):")
    print(sens.groupby("perturbation")[
        ["kendall_tau", "top1_same", "top5_overlap", "rank_gt"]]
        .mean().round(3).to_string())

    # ------------- ground-truth threshold sensitivity, both score modes
    rows = []
    for demo in preps:
        plan = load_plan(demo)
        for thr in THRESHOLDS:
            prep = prepare_pool(plan, threshold=thr,
                                max_arrangements=args.max_arrangements)
            if prep is None:
                rows.append({"plan": demo, "threshold_mm": int(thr * 1000),
                             "derivable": False})
                continue
            table = P.score_arrangements(prep["pool"], prep["grid"],
                                         weights=P.ALGO_WEIGHTS)
            for mode, weights in modes.items():
                penalty = penalties_for(table, weights)
                for variant in ("V3", "V1"):
                    if variant not in judges:
                        continue
                    order = order_for(prep["pool"], prep["grid"], penalty,
                                      judges[variant], args.lam)
                    rank = int(np.where(order == prep["gt_idx"])[0][0]) + 1
                    best_cols = all_columns(prep["pool"][order[0]])
                    gt_cols = all_columns(prep["pool"][prep["gt_idx"]])
                    rows.append({"plan": demo, "mode": mode,
                                 "threshold_mm": int(thr * 1000),
                                 "derivable": True, "variant": variant,
                                 "reachable": prep["reachable"],
                                 "rank_gt": rank, "top1": rank == 1,
                                 "top5": rank <= 5,
                                 "chamfer_m": chamfer(best_cols, gt_cols),
                                 "gt_n_columns": len(gt_cols)})
    thr_df = pd.DataFrame(rows)
    thr_df.to_csv(DATA / "evaluation_threshold.csv", index=False)
    print("\nthreshold sensitivity (mean over plans):")
    ok = thr_df[thr_df.derivable == True]                  # noqa: E712
    print(ok.groupby(["mode", "threshold_mm", "variant"])[
        ["rank_gt", "top1", "top5", "chamfer_m", "gt_n_columns"]]
        .mean().round(3).to_string())

    # ------------- score-vs-judge ranking overlap (loophole L7)
    if "V1" in judges:
        rows = []
        for demo, prep in preps.items():
            table = P.score_arrangements(prep["pool"], prep["grid"],
                                         weights=P.ALGO_WEIGHTS)
            penalty = penalties_for(table, P.ALGO_WEIGHTS)
            p_pl = judges["V1"].plausibility(prep["pool"], prep["grid"])
            rho = stats.spearmanr(penalty, 1 - p_pl).statistic
            rows.append({"plan": demo, "spearman_penalty_vs_judge":
                         float(rho)})
        ov = pd.DataFrame(rows)
        ov.to_csv(DATA / "evaluation_overlap.csv", index=False)
        print(f"\nscore-vs-ML1 ranking overlap: Spearman rho "
              f"mean {ov.spearman_penalty_vs_judge.mean():.3f}, "
              f"range [{ov.spearman_penalty_vs_judge.min():.3f}, "
              f"{ov.spearman_penalty_vs_judge.max():.3f}]")
        print("(above 0.9 would mean the judge adds nothing over the score)")


if __name__ == "__main__":
    main()
