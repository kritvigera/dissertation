# Stage 10: The quantum judge

Applies variational quantum classifiers to the task of Stage 07, using
identical examples and labels, and compares them against classical models
given the same information. This is the judge used by pipeline variant V4.

## What is and is not claimed

This is a **feasibility comparison**. Four simulated quantum bits cannot
support a claim of quantum advantage in either direction, and none is made.

Two features of the design make the comparison meaningful:

- **Capacity-matched controls.** Logistic regression on the *same four*
  selected measurements, the *same* training rows and the *same* test set as
  the quantum circuits. Comparing a four-bit circuit against a large
  ensemble trained on everything would be unfair to the circuit; comparing
  it against nothing would tell the reader nothing. Gradient boosting is
  also reported, on the same four measurements and on all six, to show what
  additional capacity and additional information are worth.
- **A training-set size sweep on one fixed test set.** Every model is fitted
  at 400, 2,000, 8,000 and about 19,900 training rows and scored on the same
  held-out plans. Holding the test set fixed is essential: an earlier version
  of this experiment measured the small-sample models on a small test set and
  the full-corpus models on a larger one, which confounded training volume
  with test difficulty and produced a misleading result.

## Method

The task and labels are those of Stages 07 and 11: real derived arrangements
against arrangements altered in geometry space, so that nothing about the
data differs between the judges.

The test set is held out first, by plan. The four most informative
measurements are then selected on the smallest training set and scaled into
an angle range, so that no training size gains an advantage from a better
choice of measurements. Each measurement becomes the rotation angle of one
quantum bit; the bits are entangled and rotated by trainable amounts, and one
is measured to produce the output. Training adjusts those rotations in the
same way any neural network is trained.

Two circuit designs are used: a standard variational classifier with two
entangling layers, and one that re-introduces the input data at every layer,
which is known to make small circuits more expressive. Both run on a
state-vector simulator, which reproduces an ideal quantum computer with no
noise; real hardware would only be harder.

Reaching the full corpus at all required one implementation change. The
circuit is now evaluated on a whole batch of rows in a single call, using
parameter broadcasting, rather than once per row in a Python loop. That is
about 160 times faster and is what previously confined this stage to a few
hundred examples.

## Results

ROC AUC on one fixed held-out test set, by the number of training rows
(`data/quantum_comparison.csv`):

| Model | 400 | 2,000 | 8,000 | 19,936 | Gain |
|---|---|---|---|---|---|
| Boosting, all 6 measurements | 0.790 | 0.841 | 0.860 | **0.865** | **+0.075** |
| Boosting, same 4 measurements | 0.705 | 0.753 | 0.779 | 0.789 | **+0.084** |
| Logistic, same 4 (capacity-matched) | 0.747 | 0.749 | 0.746 | 0.746 | −0.001 |
| VQC, one encoding, 2 layers | 0.741 | 0.751 | 0.734 | 0.748 | +0.007 |
| Data re-uploading, 6 layers | 0.741 | 0.761 | 0.756 | 0.746 | +0.005 |

Training time at the largest size: 40 s for the plain circuit and 251 s for
the re-uploading circuit, against 3 to 5 s for the tree ensembles and
under a second for logistic regression.

Three conclusions follow, and the third is the one worth stating in a
defence.

**Given the same information, the circuits match their capacity-matched
control.** At every size the two quantum models sit within about 0.015 of
logistic regression on the same four measurements, and at 2,000 rows they
are slightly ahead. A four-qubit circuit can learn this task.

**The circuits gain almost nothing from more data.** Fifty times more
training data moves them by +0.007 and +0.005, which is within run-to-run
variation. The capacity-matched linear control behaves identically
(−0.001), while the tree ensembles convert the same additional data into
+0.075 and +0.084.

**The ceiling is capacity, not data.** Because the circuits improve like a
linear model rather than like a flexible one, the limit on this task is
the expressive capacity of four qubits and two entangling layers, not the
number of examples available. This distinction could not be drawn from a
single small-sample run, and it is the reason the experiment sweeps the
training size rather than reporting one figure.

The comparison also requires that the test set be held fixed. An earlier
version of this experiment measured small-sample models on a small test
set and full-corpus models on a larger one; the circuits then appeared to
*degrade* with more data, when in fact the larger test set was simply
harder. The capacity-matched control fell by the same amount, which is
what exposed the error.

## The judge used by the pipeline

Variant V4 uses a plain two-layer circuit trained on the canonical 80 %
training partition (`data/train_test_split.csv`), so that all three
judges see identical data. Since the sweep shows the extra rows buy the
circuit almost nothing, this costs about fifty seconds at start-up and
buys a comparison that cannot be dismissed on the grounds that the
quantum arm saw less data than the others.

## A cautionary result retained deliberately

On an earlier version of the labels, which placed a column at every grid
crossing, this stage appeared to show a quantum ensemble matching a strong
classical model. When the labels were corrected the finding reversed
completely, and the ensemble became the weakest model in the table. The
circuits had not changed; the task had. The episode is retained in the
write-up because it demonstrates how readily comparisons of this kind can
produce misleadingly favourable results, and it is the reason the controls,
rather than the circuits, constitute this stage's contribution.

## Files

- `quantum_ml.ipynb` — the method, the results, and the limits of the claim
- `train_quantum.py` — the experiment
- `data/quantum_comparison.csv`
