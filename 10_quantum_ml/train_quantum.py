"""Stage 10: the quantum judge, trained at two data scales.

The task and labels are those of the feature judge (Stage 07): real derived
arrangements against arrangements altered in geometry space. Nothing about
the data differs between the judges, which is what makes the four-variant
comparison a comparison of methods.

The experiment sweeps the size of the training set, so that the effect of
data volume can be separated from the effect of the models themselves.
One test set is held out first and every model is scored on exactly that
set, which is what makes the sizes comparable: measuring a small-sample
model on a small test set and a full-corpus model on a large one would
confound training volume with test difficulty.

  400 rows      the regime an ideal quantum device could plausibly handle,
                and the regime in which published variational-classifier
                demonstrations sit
  2,000 rows
  8,000 rows
  all rows      about 19,900. Reachable only on a simulator, and only
                because the circuit is evaluated on a whole batch at once
                rather than row by row

At each scale the circuits are compared against classical controls given
exactly the same information:

  capacity-matched   logistic regression on the same four measurements,
                     the same rows and the same split. A four-qubit circuit
                     compared against a large ensemble trained on
                     everything would be judged unfairly; compared against
                     nothing it would tell the reader nothing.
  full-capacity      gradient boosting on all six measurements, on the same
                     rows, so that the value of the two measurements the
                     circuits cannot accept is visible.

Writes data/quantum_comparison.csv.

Run:  .venv/bin/python 10_quantum_ml/train_quantum.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "09_full_pipeline"))
import pipeline as P                                        # noqa: E402

from sklearn.ensemble import GradientBoostingClassifier      # noqa: E402
from sklearn.feature_selection import f_classif              # noqa: E402
from sklearn.linear_model import LogisticRegression          # noqa: E402
from sklearn.metrics import (average_precision_score,        # noqa: E402
                             roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit         # noqa: E402
from sklearn.preprocessing import MinMaxScaler                # noqa: E402

DATA = HERE.parent / "data"
SEED = 42
N_QUBITS = 4
BATCH = 256
SIZES = (400, 2000, 8000, 10 ** 9)   # the last is clipped to the pool


def judge_features(seed=SEED):
    """The shared judge dataset, as measurements."""
    corpus = P.load_judge_corpus()
    rows = P.build_judge_rows(corpus, seed)
    feats, labels, groups = [], [], []
    for points, label, group, source, kind in rows:
        f = P.layout_features(points)
        if f is None:
            continue
        feats.append(f)
        labels.append(label)
        groups.append(group)
    return (pd.DataFrame(feats)[P.LAYOUT_FEATURES],
            np.asarray(labels, float), np.asarray(groups))


def train_circuit(circuit, shape, X_tr, y_tr, epochs, lr, seed=SEED,
                  log=print, label=""):
    """Fit one circuit by minibatch Adam.

    The circuit is evaluated on a whole minibatch in one call, using
    PennyLane's parameter broadcasting. Evaluating row by row, as an earlier
    version did, costs about 160 times more and is what previously confined
    this stage to a few hundred rows.
    """
    import pennylane as qml
    from pennylane import numpy as pnp

    weights = pnp.array(0.1 * np.random.default_rng(seed).standard_normal(
        shape), requires_grad=True)
    bias = pnp.array(0.0, requires_grad=True)
    opt = qml.AdamOptimizer(lr)
    Xp = pnp.array(X_tr, requires_grad=False)
    yp = pnp.array(y_tr, requires_grad=False)

    def loss(w, b, xb, yb):
        p = 1 / (1 + pnp.exp(-4 * (circuit(w, xb) + b)))
        return -pnp.mean(yb * pnp.log(p + 1e-9) +
                         (1 - yb) * pnp.log(1 - p + 1e-9))

    idx = np.arange(len(Xp))
    for epoch in range(epochs):
        np.random.default_rng(seed + epoch).shuffle(idx)
        for s in range(0, len(idx), BATCH):
            b = idx[s:s + BATCH]
            weights, bias = opt.step(
                lambda w, bb: loss(w, bb, Xp[b], yp[b]), weights, bias)
        if log and (epoch + 1) % 10 == 0:
            log(f"    {label} epoch {epoch + 1}/{epochs} "
                f"loss {float(loss(weights, bias, Xp[idx[:BATCH]], yp[idx[:BATCH]])):.4f}")

    def predict(X_):
        out = []
        for s in range(0, len(X_), 4096):        # chunked, to bound memory
            z = np.asarray(circuit(weights, pnp.array(
                X_[s:s + 4096], requires_grad=False)), dtype=float)
            out.append(z)
        z = np.concatenate(out)
        return 1 / (1 + np.exp(-4 * (z + float(bias))))
    return predict


def build_circuits():
    import pennylane as qml
    dev = qml.device("default.qubit", wires=N_QUBITS)

    @qml.qnode(dev, interface="autograd", diff_method="backprop")
    def plain(weights, x):
        qml.AngleEmbedding(x, wires=range(N_QUBITS))
        qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
        return qml.expval(qml.PauliZ(0))

    @qml.qnode(dev, interface="autograd", diff_method="backprop")
    def reupload(weights, x):
        for layer in range(weights.shape[0]):
            for q in range(N_QUBITS):
                qml.RY(weights[layer, q, 0] * x[..., q] + weights[layer, q, 1],
                       wires=q)
                qml.RZ(weights[layer, q, 2] * x[..., q] + weights[layer, q, 3],
                       wires=q)
            for q in range(N_QUBITS):
                qml.CNOT(wires=[q, (q + 1) % N_QUBITS])
        return qml.expval(qml.PauliZ(0))

    return plain, reupload


def run_size(n_rows, X, y, tr_all, te, top4, scaler, log=print):
    """Fit every model on the first n_rows of the training pool.

    The test set and the selected measurements are fixed before this
    function is called, so results at different sizes are comparable.
    """
    import pennylane as qml

    tr = tr_all[:n_rows]
    A_tr = scaler.transform(X.iloc[tr][top4])
    A_te = np.clip(scaler.transform(X.iloc[te][top4]), 0, np.pi)
    label = "all" if n_rows >= len(tr_all) else str(n_rows)
    log(f"\n[{len(tr)} rows] fitting")

    out = []

    def record(model, kind, p_te, seconds):
        out.append({"train_rows": len(tr), "size_label": label,
                    "model": model, "kind": kind, "test_rows": len(te),
                    "accuracy": float(((p_te > 0.5) == y[te]).mean()),
                    "roc_auc": float(roc_auc_score(y[te], p_te)),
                    "pr_auc": float(average_precision_score(y[te], p_te)),
                    "train_seconds": round(seconds, 1)})
        log(f"  {model}: ROC AUC {out[-1]['roc_auc']:.3f}, "
            f"PR AUC {out[-1]['pr_auc']:.3f} ({seconds:.0f}s)")

    plain, reupload = build_circuits()

    t0 = time.time()
    predict = train_circuit(plain, qml.StronglyEntanglingLayers.shape(2, N_QUBITS),
                            A_tr, y[tr], epochs=40, lr=0.15, log=None)
    record("VQC, one encoding, 2 layers", "quantum", predict(A_te),
           time.time() - t0)

    t0 = time.time()
    predict = train_circuit(reupload, (6, N_QUBITS, 4), A_tr, y[tr],
                            epochs=70, lr=0.10, log=None)
    record("data re-uploading, 6 layers", "quantum", predict(A_te),
           time.time() - t0)

    t0 = time.time()
    lg = LogisticRegression(max_iter=2000).fit(A_tr, y[tr])
    record("logistic, same 4 measurements (capacity-matched)", "classical",
           lg.predict_proba(A_te)[:, 1], time.time() - t0)

    t0 = time.time()
    g4 = GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                    random_state=SEED).fit(A_tr, y[tr])
    record("boosting, same 4 measurements", "classical",
           g4.predict_proba(A_te)[:, 1], time.time() - t0)

    t0 = time.time()
    g6 = GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                    random_state=SEED
                                    ).fit(X.iloc[tr], y[tr])
    record("boosting, all 6 measurements", "classical",
           g6.predict_proba(X.iloc[te])[:, 1], time.time() - t0)
    return out


def main():
    X, y, groups = judge_features()
    print(f"shared judge dataset: {len(X)} rows over "
          f"{len(np.unique(groups))} plans")

    # One test set, held out by plan, used for every training size.
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
    tr_all, te = next(gss.split(X, y, groups=groups))
    rng = np.random.default_rng(SEED)
    tr_all = rng.permutation(tr_all)
    print(f"held-out test set: {len(te)} rows; training pool {len(tr_all)} rows")

    # Measurements are selected and scaled on the smallest training set, so
    # that no size gains an advantage from a better choice of measurements.
    warm = tr_all[:SIZES[0]]
    scores, _ = f_classif(X.iloc[warm], y[warm])
    top4 = [P.LAYOUT_FEATURES[i]
            for i in np.argsort(scores)[::-1][:N_QUBITS]]
    scaler = MinMaxScaler((0, np.pi)).fit(X.iloc[warm][top4])
    print(f"selected measurements: {top4}")

    rows = []
    for n in SIZES:
        rows += run_size(min(n, len(tr_all)), X, y, tr_all, te, top4, scaler)

    out = pd.DataFrame(rows)
    out.to_csv(DATA / "quantum_comparison.csv", index=False)
    print("\n" + out[["train_rows", "model", "kind", "roc_auc", "pr_auc",
                      "train_seconds"]].round(3).to_string(index=False))

    print("\nROC AUC by training-set size (one common test set):")
    piv = out.pivot_table(index=["kind", "model"], columns="train_rows",
                          values="roc_auc")
    piv["gain"] = piv[piv.columns[-1]] - piv[piv.columns[0]]
    print(piv.round(3).to_string())


if __name__ == "__main__":
    main()
