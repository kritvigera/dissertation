"""Emit every report table as LaTeX, from the committed results tables."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "report" / "tables"
OUT.mkdir(parents=True, exist_ok=True)


def write(name, latex):
    (OUT / f"{name}.tex").write_text(latex)
    print(f"  {name}.tex")


def corpus():
    f = pd.read_csv(DATA / "features_all.csv")
    rows = []
    for src, fmt, units, conf in (
            ("ResPlan", "pickled polygons", "$\\approx$ decimetres", "none"),
            ("Swiss Dwellings", "polygons (CSV)", "metres", "verified"),
            ("CubiCasa5K", "SVG drawings", "drawing pixels", "relative")):
        sub = f[f.source == src.split()[0]]
        rows.append((src, len(sub), fmt, units, conf,
                     f"{sub.n_columns.median():.0f}",
                     f"{sub.regularity.median():.3f}"))
    body = "\n".join(f"{a} & {b:,} & {c} & {d} & {e} & {g} & {h} \\\\"
                     for a, b, c, d, e, g, h in rows)
    write("corpus", f"""\\setlength{{\\tabcolsep}}{{4pt}}
\\begin{{tabular}}{{@{{}}lrllrrr@{{}}}}
\\toprule
 & & & & & Median & Median \\\\
Source & Plans & Native format & Units drawn & Labels & cols. & reg. \\\\
\\midrule
{body}
\\midrule
\\textbf{{Total}} & \\textbf{{{len(f):,}}} & & & & & \\\\
\\bottomrule
\\end{{tabular}}""")


def threshold_corpus():
    t = pd.read_csv(DATA / "threshold_sweep.csv")
    sw = t[t["mode"] == "absolute"]
    body = "\n".join(
        f"{r.threshold*1000:.0f} & {r.plans_kept:,} & {r.median_columns:.0f} \\\\"
        for r in sw.itertuples())
    write("threshold_corpus", f"""\\begin{{tabular}}{{rrr}}
\\toprule
Threshold (mm) & Plans with labels & Median columns \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}""")


def weights():
    w = json.loads((DATA / "learned_weights.json").read_text())
    names = {"t_density": "Density", "t_span_demand": "Span demand",
             "t_irregularity": "Irregularity", "t_off_wall": "Off-wall",
             "t_repetition": "Repetition"}
    body = "\n".join(
        f"{names[k]} & 1.00 & ${w['weights'][k]:+.2f}$ & "
        f"{w['coefficient_std_over_splits'][k]:.2f} \\\\" for k in names)
    write("weights", f"""\\begin{{tabular}}{{lrrr}}
\\toprule
Measure & Fixed weight & Learned weight & S.d.\\ over splits \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}""")


def ml1():
    m = pd.read_csv(DATA / "ml1_results.csv")
    p = m[(m.experiment == "grouped-split") & (m.test == "pooled")]
    st = m[m.experiment == "stump-best"]
    ps = m[(m.experiment == "grouped-split") & (m.test.isin(["Swiss", "CubiCasa"]))]
    lines = [f"Feature judge, five grouped splits & {p.roc_auc.mean():.3f} $\\pm$ "
             f"{p.roc_auc.std():.3f} & {p.pr_auc.mean():.3f} $\\pm$ {p.pr_auc.std():.3f} \\\\",
             f"Best single-measurement threshold & {st.roc_auc.mean():.3f} $\\pm$ "
             f"{st.roc_auc.std():.3f} & --- \\\\", "\\midrule"]
    for s in ("Swiss", "CubiCasa"):
        q = ps[ps.test == s]
        lines.append(f"\\quad tested on {s} only & {q.roc_auc.mean():.3f} & "
                     f"{q.pr_auc.mean():.3f} \\\\")
    lines.append("\\midrule")
    for r in m[m.experiment == "leave-one-source-out"].itertuples():
        lines.append(f"Trained {r.train}, tested {r.test} & {r.roc_auc:.3f} & "
                     f"{r.pr_auc:.3f} \\\\")
    lines.append("\\midrule")
    for r in m[m.experiment == "per-negative-kind"].itertuples():
        lines.append(f"Against {r.test} examples only & {r.roc_auc:.3f} & "
                     f"{r.pr_auc:.3f} \\\\")
    write("ml1", f"""\\begin{{tabular}}{{lrr}}
\\toprule
Experiment & ROC AUC & PR AUC \\\\
\\midrule
{chr(10).join(lines)}
\\bottomrule
\\end{{tabular}}""")


def ml2():
    m = pd.read_csv(DATA / "ml2_results.csv")
    pl = m[m.test == "pooled"].groupby("model")[["roc_auc", "pr_auc"]].agg(["mean", "std"])
    label = {"ML2 CNN on pictures": "Image judge, walls and columns",
             "ML2 CNN, columns only": "Image judge, walls layer blanked",
             "ML1 boosting on features (paired)": "Feature judge, paired on the same rows"}
    lines = [f"{label[k]} & {pl.loc[k,('roc_auc','mean')]:.3f} $\\pm$ "
             f"{pl.loc[k,('roc_auc','std')]:.3f} & {pl.loc[k,('pr_auc','mean')]:.3f} $\\pm$ "
             f"{pl.loc[k,('pr_auc','std')]:.3f} \\\\"
             for k in ("ML2 CNN on pictures", "ML2 CNN, columns only",
                       "ML1 boosting on features (paired)")]
    lines.append("\\midrule")
    for r in m[m.train.isin(["Swiss", "CubiCasa"])].itertuples():
        if r.model in label:
            lines.append(f"{label[r.model]}, {r.train}$\\rightarrow${r.test} & "
                         f"{r.roc_auc:.3f} & {r.pr_auc:.3f} \\\\")
    write("ml2", f"""\\begin{{tabular}}{{lrr}}
\\toprule
Model & ROC AUC & PR AUC \\\\
\\midrule
{chr(10).join(lines)}
\\bottomrule
\\end{{tabular}}""")


def quantum():
    q = pd.read_csv(DATA / "quantum_comparison.csv")
    sizes = sorted(q.train_rows.unique())
    order = ["boosting, all 6 measurements", "boosting, same 4 measurements",
             "logistic, same 4 measurements (capacity-matched)",
             "VQC, one encoding, 2 layers", "data re-uploading, 6 layers"]
    nice = {order[0]: "Boosting, all 6 measurements",
            order[1]: "Boosting, same 4 measurements",
            order[2]: "Logistic, same 4 (capacity-matched)",
            order[3]: "VQC, one encoding, 2 layers",
            order[4]: "Data re-uploading, 6 layers"}
    piv = q.pivot_table(index="model", columns="train_rows", values="roc_auc")
    sec = q.pivot_table(index="model", columns="train_rows", values="train_seconds")
    lines = []
    for m in order:
        gain = piv.loc[m, sizes[-1]] - piv.loc[m, sizes[0]]
        cells = " & ".join(f"{piv.loc[m,s]:.3f}" for s in sizes)
        kind = "Q" if ("VQC" in m or "re-upload" in m) else "C"
        lines.append(f"{nice[m]} & {kind} & {cells} & ${gain:+.3f}$ & "
                     f"{sec.loc[m, sizes[-1]]:.0f} \\\\")
        if m == order[2]:
            lines.append("\\midrule")
    hdr = " & ".join(f"{s:,}" for s in sizes)
    write("quantum", f"""\\begin{{tabular}}{{llrrrrrr}}
\\toprule
Model & & \\multicolumn{{4}}{{c}}{{Training rows}} & Gain & Time (s) \\\\
\\cmidrule(lr){{3-6}}
 & & {hdr} & & \\\\
\\midrule
{chr(10).join(lines)}
\\bottomrule
\\end{{tabular}}""")


def accuracy_and_tests():
    a = pd.read_csv(DATA / "evaluation_accuracy.csv")
    g = a.groupby(["mode", "variant"])
    vlab = {"V1": "V1 (feature judge)", "V2": "V2 (image judge)",
            "V3": "V3 (score only, baseline)", "V4": "V4 (quantum judge)"}
    lines = []
    for mode, mlab in (("algo", "Fixed weights"), ("learned", "Learned weights")):
        lines.append(f"\\multicolumn{{7}}{{l}}{{\\emph{{{mlab}}}}} \\\\")
        for v in ("V1", "V2", "V3", "V4"):
            r = g.get_group((mode, v))
            lines.append(f"\\quad {vlab[v]} & {r.rank_gt.mean():.1f} & "
                         f"{r.rank_gt.median():.0f} & {r.top1.mean():.3f} & "
                         f"{r.top5.mean():.3f} & {r.chamfer_m.mean():.3f} & "
                         f"{r.iou.mean():.3f} \\\\")
        if mode == "algo":
            lines.append("\\midrule")
    write("accuracy", f"""\\begin{{tabular}}{{lrrrrrr}}
\\toprule
Variant & Mean rank & Median rank & Top-1 & Top-5 & Distance (m) & Overlap \\\\
\\midrule
{chr(10).join(lines)}
\\bottomrule
\\end{{tabular}}""")

    t = pd.read_csv(DATA / "evaluation_tests.csv")
    mn = {"rank_gt": "Rank", "chamfer_m": "Distance", "iou": "Overlap"}
    sig = t[(t.wilcoxon_p < 1.0) & (t["mode"] == "learned")]
    body = "\n".join(f"{r.pair} & {mn[r.metric]} & {r.mean_a:.3f} & {r.mean_b:.3f} & "
                     f"{r.wilcoxon_p:.3f} & {r.sign_p:.3f} \\\\" for r in sig.itertuples())
    write("tests", f"""\\begin{{tabular}}{{llrrrr}}
\\toprule
Comparison & Metric & Mean A & Mean B & Wilcoxon $p$ & Sign $p$ \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}""")


def sensitivity_lambda_threshold():
    short = {"t_density": "Density", "t_irregularity": "Irregularity",
             "t_off_wall": "Off-wall", "t_repetition": "Repetition",
             "t_span_demand": "Span demand"}

    def label(idx):
        if idx == "nominal":
            return "Nominal (unperturbed)"
        base, _, suffix = idx.rpartition(" ")
        return f"{short.get(base, base)}, {suffix.replace('dropped', 'removed')}"

    s = pd.read_csv(DATA / "evaluation_weights_sensitivity.csv").groupby(
        "perturbation")[["kendall_tau", "top1_same", "top5_overlap", "rank_gt"]].mean()
    body = "\n".join(f"{label(i)} & {r.kendall_tau:.3f} & {r.top1_same:.3f} & "
                     f"{r.top5_overlap:.3f} & {r.rank_gt:.0f} \\\\" for i, r in s.iterrows())
    write("sensitivity", f"""\\setlength{{\\tabcolsep}}{{5pt}}
\\begin{{tabular}}{{@{{}}lrrrr@{{}}}}
\\toprule
Perturbation & Kendall $\\tau$ & Top-1 kept & Top-5 kept & Rank \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}""")

    lam = pd.read_csv(DATA / "evaluation_lambda.csv").groupby(
        ["mode", "lambda"])[["rank_gt", "top1", "top5"]].mean()
    lines = []
    for mode in ("algo", "learned"):
        lines.append(f"\\multicolumn{{4}}{{l}}{{\\emph{{"
                     f"{'Fixed' if mode == 'algo' else 'Learned'} weights}}}} \\\\")
        for (m, lv), r in lam.iterrows():
            if m == mode:
                lines.append(f"\\quad $\\lambda = {lv}$ & {r.rank_gt:.1f} & "
                             f"{r.top1:.3f} & {r.top5:.3f} \\\\")
        if mode == "algo":
            lines.append("\\midrule")
    write("lambda", f"""\\begin{{tabular}}{{lrrr}}
\\toprule
Judge weight & Mean rank & Top-1 & Top-5 \\\\
\\midrule
{chr(10).join(lines)}
\\bottomrule
\\end{{tabular}}""")

    thr = pd.read_csv(DATA / "evaluation_threshold.csv")
    ok = thr[thr.derivable == True]                                # noqa: E712
    tg = ok.groupby(["mode", "threshold_mm", "variant"])[
        ["rank_gt", "top1", "top5", "chamfer_m", "gt_n_columns"]].mean()
    lines = []
    for mode in ("algo", "learned"):
        lines.append(f"\\multicolumn{{7}}{{l}}{{\\emph{{"
                     f"{'Fixed' if mode == 'algo' else 'Learned'} weights}}}} \\\\")
        for (m, th, v), r in tg.iterrows():
            if m == mode:
                lines.append(f"\\quad {th} & {v} & {r.gt_n_columns:.0f} & "
                             f"{r.rank_gt:.0f} & {r.top1:.3f} & {r.top5:.3f} & "
                             f"{r.chamfer_m:.3f} \\\\")
        if mode == "algo":
            lines.append("\\midrule")
    write("threshold_accuracy", f"""\\begin{{tabular}}{{rlrrrrr}}
\\toprule
Threshold (mm) & Variant & Label cols. & Mean rank & Top-1 & Top-5 & Distance (m) \\\\
\\midrule
{chr(10).join(lines)}
\\bottomrule
\\end{{tabular}}""")


def demo_plans():
    dm = pd.read_csv(DATA / "demo_plans" / "index.csv")
    a = pd.read_csv(DATA / "evaluation_accuracy.csv")
    acc = a[(a.variant == "V3") & (a["mode"] == "learned")].set_index("plan")
    body = "\n".join(
        f"{r.demo.replace('_', chr(92)+'_')} & {r.source_id} & "
        f"{r.footprint_x_m:.1f}$\\times${r.footprint_y_m:.1f} & "
        f"{r.n_lines_x}$\\times${r.n_lines_y} & {r.n_columns} & "
        f"{r.n_annotated_columns} & {acc.loc[r.demo,'n_candidates']:,} & "
        f"{acc.loc[r.demo,'rank_gt']} \\\\" for r in dm.itertuples())
    write("demo_plans", f"""\\begin{{tabular}}{{llrrrrrr}}
\\toprule
Plan & Source id & Footprint (m) & Grid & Cols. & Marked & Candidates & Rank (V3) \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}""")


if __name__ == "__main__":
    corpus()
    threshold_corpus()
    weights()
    ml1()
    ml2()
    quantum()
    accuracy_and_tests()
    sensitivity_lambda_threshold()
    demo_plans()
