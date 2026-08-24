"""Redraw the report's figures from the committed data.

These are the same data as the repository figures in figures/, drawn
without in-figure titles: in a formal report the caption carries the
explanation. Written to report/figures/, which precedes ../figures/ in the
LaTeX graphics path, so the remaining figures fall through to the
repository versions.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                            # noqa: E402
import numpy as np                                          # noqa: E402
import pandas as pd                                         # noqa: E402
from matplotlib.patches import Patch                        # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150})
TERMS = ["t_density", "t_span_demand", "t_irregularity", "t_off_wall",
         "t_repetition"]


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}")


def weights():
    w = json.loads((DATA / "learned_weights.json").read_text())["weights"]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    x = np.arange(len(TERMS))
    ax.bar(x - 0.2, [1.0] * len(TERMS), 0.4, color="#9aa5b1",
           label="Fixed (all equal)")
    ax.bar(x + 0.2, [w[t] for t in TERMS], 0.4, color="#2563eb",
           label="Learned by ranking")
    ax.axhline(0, color="#1c2430", lw=0.8)
    ax.set_xticks(x, ["density", "span demand", "irregularity", "off-wall",
                      "repetition"])
    ax.set_ylabel("Weight")
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, "06_weights.png")


def image_judge():
    m = pd.read_csv(DATA / "ml2_results.csv")
    pl = m[m.test == "pooled"].groupby("model").roc_auc.agg(["mean", "std"])
    order = ["ML1 boosting on features (paired)", "ML2 CNN, columns only",
             "ML2 CNN on pictures"]
    labels = ["Feature judge\n(6 measurements)", "Image judge\n(columns only)",
              "Image judge\n(walls + columns)"]
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    ax.bar(labels, [pl.loc[m_, "mean"] for m_ in order],
           yerr=[pl.loc[m_, "std"] for m_ in order], capsize=4,
           color=["#9aa5b1", "#60a5fa", "#2563eb"])
    ax.set_ylim(0.5, 1.02)
    ax.set_ylabel("ROC AUC")
    for i, m_ in enumerate(order):
        ax.text(i, pl.loc[m_, "mean"] + 0.02, f"{pl.loc[m_,'mean']:.3f}",
                ha="center", fontsize=8)
    fig.tight_layout()
    save(fig, "11_cnn_vs_features.png")


def quantum():
    q = pd.read_csv(DATA / "quantum_comparison.csv")
    style = {"quantum": dict(marker="o", ls="-"),
             "classical": dict(marker="s", ls="--")}
    colour = {"VQC, one encoding, 2 layers": "#7c3aed",
              "data re-uploading, 6 layers": "#a855f7",
              "logistic, same 4 measurements (capacity-matched)": "#6b7280",
              "boosting, same 4 measurements": "#60a5fa",
              "boosting, all 6 measurements": "#2563eb"}
    nice = {"VQC, one encoding, 2 layers": "VQC, 2 layers (quantum)",
            "data re-uploading, 6 layers": "Re-uploading, 6 layers (quantum)",
            "logistic, same 4 measurements (capacity-matched)":
                "Logistic, 4 meas. (capacity-matched)",
            "boosting, same 4 measurements": "Boosting, 4 meas.",
            "boosting, all 6 measurements": "Boosting, 6 meas."}
    order = ["boosting, all 6 measurements", "boosting, same 4 measurements",
             "logistic, same 4 measurements (capacity-matched)",
             "VQC, one encoding, 2 layers", "data re-uploading, 6 layers"]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    handles = []
    for model in order:
        grp = q[q.model == model].sort_values("train_rows")
        h, = ax.plot(grp.train_rows, grp.roc_auc, color=colour[model],
                     label=nice[model], **style[grp.kind.iloc[0]])
        handles.append(h)
    ax.set_xscale("log")
    ax.set_xlabel("Training rows (logarithmic scale)")
    ax.set_ylabel("ROC AUC on the fixed test set")
    ax.set_ylim(0.68, 0.90)
    ax.legend(handles=handles, fontsize=7.5, frameon=False, ncol=2,
              loc="upper center", bbox_to_anchor=(0.5, -0.28))
    fig.tight_layout()
    save(fig, "10_quantum_vs_classical.png")


def accuracy():
    a = pd.read_csv(DATA / "evaluation_accuracy.csv")
    vs = ["V1", "V2", "V3", "V4"]
    cols = {"V1": "#2563eb", "V2": "#0ea5e9", "V3": "#6b7280", "V4": "#7c3aed"}
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.0))
    for ax, metric, ylab in ((axes[0], "chamfer_m", "Distance to label (m)"),
                             (axes[1], "iou", "Column overlap")):
        for i, mode in enumerate(("algo", "learned")):
            sub = a[a["mode"] == mode].groupby("variant")[metric].mean()
            ax.bar(np.arange(4) + (i - 0.5) * 0.38, [sub[v] for v in vs], 0.36,
                   color=[cols[v] for v in vs],
                   alpha=1.0 if mode == "learned" else 0.4)
        ax.set_xticks(np.arange(4), vs)
        ax.set_ylabel(ylab)
    axes[0].legend(handles=[Patch(facecolor="#6b7280", alpha=0.4,
                                  label="Fixed weights"),
                            Patch(facecolor="#6b7280", label="Learned weights")],
                   fontsize=7.5, frameon=False, loc="upper right")
    fig.tight_layout()
    save(fig, "12_accuracy.png")


def regularity():
    f = pd.read_csv(DATA / "features_all.csv")
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    for src, c in (("Swiss", "#2563eb"), ("CubiCasa", "#0ea5e9"),
                   ("ResPlan", "#9aa5b1")):
        v = f[f.source == src].regularity.dropna()
        ax.hist(v, bins=50, histtype="step", density=True, lw=1.6, color=c,
                label=f"{src} (median {v.median():.3f})")
    ax.set_xlabel("Regularity  (1 = every bay identical)")
    ax.set_ylabel("Density")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, "03_regularity.png")


if __name__ == "__main__":
    weights()
    image_judge()
    quantum()
    accuracy()
    regularity()
