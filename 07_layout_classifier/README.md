# Stage 07: The feature judge

Estimates the probability that a candidate arrangement is plausible, using
six numerical descriptions of it. This is the judge used by pipeline
variant V1, which re-orders the fifteen best-scoring candidates by
combining the score with the judge's opinion.

## Correcting the training examples

A judge learns from examples: real arrangements labelled plausible,
implausible ones labelled otherwise. Its honesty depends entirely on how
the implausible examples are made.

The previous version made them by editing the *measurements*: it multiplied
the two bay-variability figures, overwrote the interior-column proportion,
stretched the aspect ratio, and recalculated the regularity measure derived
from them. Those are four of the six numbers the model then read. The task
therefore reduced to detecting which measurements had been altered, and a
single-threshold rule on one measurement recovered most of the reported
performance. That defect was documented at the time; because the judge is
now a contribution rather than a component, it has been corrected.

Implausible examples are now created by moving the **column positions**,
before any measurement is taken:

| Type | Construction | What it tests |
|---|---|---|
| Transplant | another plan's real arrangement, rescaled onto this footprint | whether the arrangement suits *this* building |
| Jitter | every column displaced by a random amount | whether columns align into lines at all |
| Shear | the arrangement skewed so that lines are no longer perpendicular | orthogonality |
| Dropout | 30% of columns removed and some survivors displaced | completeness of the load path |

## Data and model

Positives are the derived arrangements of the 12,460 usable plans, with the
seven demonstration plans removed before anything is constructed. The model
is a gradient-boosted decision ensemble over six measurements chosen to be
independent of scale: aspect ratio, bay variability in each direction,
regularity, the proportion of interior columns, and the ratio of mean bay
sizes. The restriction to scale-independent measurements is forced by the
corpus, whose three sources are drawn in three different unit systems.

The same examples and labels are used by the image judge (Stage 12) and the
quantum judge (Stage 11). That shared basis is what makes the four-variant
comparison a comparison of methods rather than of datasets.

## Results

Splits are grouped by plan, so that a plan and its altered version can
never fall on opposite sides, and are repeated over five random seeds.
Results are in `data/ml1_results.csv`.

| | ROC AUC | Precision-recall AUC |
|---|---|---|
| Feature judge, five grouped splits | **0.862 ± 0.006** | 0.831 ± 0.008 |
| Best single-measurement threshold (regularity) | 0.679 ± 0.008 | — |
| Trained on Swiss, tested on CubiCasa | 0.673 | 0.608 |
| Trained on CubiCasa, tested on Swiss | 0.805 | 0.727 |
| Against transplanted arrangements only | 0.668 | — |
| Against jitter, shear and dropout | 0.960 / 0.956 / 0.881 | — |

Four observations:

- The headline figure fell from 0.947 to 0.862 when the training examples
  were corrected. Roughly a tenth of the previous figure came from the
  model recognising its own example generator. The reduction is reported as
  a finding rather than adjusted away.
- No single measurement approaches the full model, so the remaining signal
  genuinely requires several measurements together.
- Transplanted arrangements are the hardest class, since they are real
  arrangements merely placed on the wrong building. They are also the class
  closest to the judge's actual task, so the headline figure should be read
  alongside them.
- Transfer between sources is asymmetric: a model trained on Swiss
  drawings generalises less well than the reverse. Cross-source claims
  therefore rest on the pooled model.

The two classes are balanced in equal proportion by construction, so
neither figure should be read as a deployment estimate; both are reported
for completeness.

## Files

- `layout_classifier.ipynb` — the four example types drawn as figures, and
  the results
- `train_ml1.py` — the experiment: five grouped splits, single-measurement
  baselines, cross-source transfer, and difficulty by example type
- `figures/07_negatives.png`, `figures/07_model_performance.png`
