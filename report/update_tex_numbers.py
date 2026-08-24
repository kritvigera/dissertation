"""Apply the same numeric updates to the LaTeX prose that
update_docx_numbers.py applies to the docx. Tables and figures are already
regenerated from the data by make_assets.py; this touches only the numbers
quoted inside sentences of ch5 and ch6. Language is not changed.

Run:  .venv/bin/python report/update_tex_numbers.py
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
REP = ROOT / "report"


def words(n):
    return {0: "no", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
            6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
            11: "eleven", 12: "twelve"}[int(n)]


def main():
    a = pd.read_csv(DATA / "evaluation_accuracy.csv")
    tests = pd.read_csv(DATA / "evaluation_tests.csv")
    lam = pd.read_csv(DATA / "evaluation_lambda.csv")
    thr = pd.read_csv(DATA / "evaluation_threshold.csv")
    ov = pd.read_csv(DATA / "evaluation_overlap.csv")

    L = a[a["mode"] == "learned"].groupby("variant")
    v2, v3 = L.get_group("V2"), L.get_group("V3")
    lamL = lam[lam["mode"] == "learned"].groupby("lambda").rank_gt.mean()
    ok = thr[thr.derivable == True]                                # noqa: E712
    t3 = ok[(ok["mode"] == "learned") & (ok.variant == "V3")].groupby("threshold_mm")
    pmin = tests[tests["mode"] == "learned"].wilcoxon_p.min()

    v4 = L.get_group("V4")
    med3 = words(v3.rank_gt.median())
    n_v4_first = int(v4.top1.sum())
    lamJ = lam[lam["mode"] == "learned"].groupby("lambda")[
        ["top1", "top5"]].mean()
    d_t1 = round((lamJ.loc[0.0, "top1"] - lamJ.loc[1.0, "top1"]) * 7)
    d_t5 = round((lamJ.loc[1.0, "top5"] - lamJ.loc[0.0, "top5"]) * 7)

    reps = [
        ("median rank of 8, recovers it exactly\non two of the seven plans, and averages 0.246\\,m Chamfer distance with 0.749\noverlap",
         f"median rank of {v3.rank_gt.median():.0f}, recovers it exactly\non {words((v3.rank_gt == 1).sum())} of the seven plans, and averages {v3.chamfer_m.mean():.3f}\\,m Chamfer distance with {v3.iou.mean():.3f}\noverlap"),
        ("2,400 to eight", f"2,400 to {med3}"),
        ("2,395 to eight", f"2,395 to {med3}"),
        ("recovers the arrangement first on no\nplan",
         f"recovers the arrangement first on {words(n_v4_first)}\nplan"
         + ("s" if n_v4_first > 1 else "")),
        ("costs one first-place recovery, and gains one top-five placing",
         f"costs {words(abs(d_t1))} first-place "
         f"recover{'y' if abs(d_t1) == 1 else 'ies'}, and gains "
         f"{words(abs(d_t5))} top-five placing"
         + ("s" if abs(d_t5) > 1 else "")),
        ("reducing distance from 0.246 to 0.191\\,m and raising\noverlap from 0.749 to 0.790",
         f"reducing distance from {v3.chamfer_m.mean():.3f} to {v2.chamfer_m.mean():.3f}\\,m and raising\noverlap from {v3.iou.mean():.3f} to {v2.iou.mean():.3f}"),
        ("The smallest $p$-value obtained is 0.125",
         f"The smallest $p$-value obtained is {pmin:.3f}"),
        ("the smallest $p$-value is 0.125",
         f"the smallest $p$-value is {pmin:.3f}"),
        ("from 23.1 to 22.9",
         f"from {lamL.loc[0.0]:.1f} to {lamL.loc[1.0]:.1f}"),
        ("mean rank of about 25 and 23 respectively and distance of 0.235 and\n0.246\\,m",
         f"mean rank of about {t3.get_group(140).rank_gt.mean():.0f} and {t3.get_group(160).rank_gt.mean():.0f} respectively and distance of {t3.get_group(140).chamfer_m.mean():.3f} and\n{t3.get_group(160).chamfer_m.mean():.3f}\\,m"),
        ("reaching rank 99 at 180\\,mm and 214 at\n200\\,mm with distance rising to 0.616\\,m",
         f"reaching rank {t3.get_group(180).rank_gt.mean():.0f} at 180\\,mm and {t3.get_group(200).rank_gt.mean():.0f} at\n200\\,mm with distance rising to {t3.get_group(200).chamfer_m.mean():.3f}\\,m"),
        ("has a mean of 0.148 and a range of $-0.019$ to $0.348$",
         f"has a mean of {ov.spearman_penalty_vs_judge.mean():.3f} and a range of ${ov.spearman_penalty_vs_judge.min():.3f}$ to ${ov.spearman_penalty_vs_judge.max():.3f}$".replace("$-", "$-")),
    ]
    for fname in ("ch5_results.tex", "ch6_conclusion.tex", "front.tex"):
        p = REP / fname
        s = p.read_text()
        n = 0
        for old, new in reps:
            if old in s and old != new:
                s = s.replace(old, new)
                n += 1
        p.write_text(s)
        print(f"  {fname}: {n} replacements")


if __name__ == "__main__":
    main()
