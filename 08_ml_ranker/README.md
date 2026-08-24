# Stage 08: Learning the score weights

Fits the five weights of the Stage 06 score function to the corpus, by
requiring that the real arrangement of each plan scores better than
alternatives generated on that same plan.

## What this stage used to be

This folder previously held a second learned model, which narrowed a
shortlist of fifteen candidates to five. It has been retired. Under the
four-variant design its role, a second learned opinion about the same
geometry, is precisely what the judges in Stages 07, 10 and 11 provide, and
retaining both would count one source of evidence twice. The stage keeps
its real function, the learned part of the ranking, in a form whose output
is five interpretable numbers rather than an opaque model. The retired
model's published figures belong to a different task and are not compared
with anything in the present pipeline.

## Why ranking, and not the alternatives

Two other formulations were considered and rejected as unsound.

Fitting a model to reproduce the fixed score's output would be circular:
the learned score would be a copy of the thing it is meant to be compared
against.

Training on the judges' real-versus-altered examples would make the learned
score and the judge the same model applied twice, so that comparing the
pipeline with and without a judge would measure nothing.

What remains is ranking. For each of 11,098 plans, the real derived
arrangement is the positive example and up to five alternatives generated
on the same plan's grid are the negatives: the full set of crossings, and
subsets of the grid lines under both placement rules. This gives 52,724
paired comparisons. A logistic model without an intercept is fitted to the
differences between the normalised measures. Because the model is linear in
those differences, its coefficients are the weights.

One overlap is stated rather than argued away: the alternatives used here
and the altered examples used by the judges both include arrangements that
place columns at every crossing. The two sets of examples are otherwise
distinct, and this residual overlap is carried in the limitations.

## Results

| | Paired comparisons ranked correctly |
|---|---|
| Learned weights | **89.4% ± 0.6%** |
| Fixed weights (all equal), same held-out plans | 74.7% |

| Weight | density | span demand | irregularity | off-wall | repetition |
|---|---|---|---|---|---|
| Value (± deviation over splits) | −0.37 ± 0.03 | +0.64 ± 0.17 | −0.49 ± 0.06 | **+2.79 ± 0.09** | +0.96 ± 0.01 |

Span demand and repetition carry the signs that design reasoning predicts:
real arrangements have shorter loaded spans and more repeated bays than the
alternatives. Off-wall carries the largest weight, though partly by
construction, since the derived labels place columns inside walls by
definition. Density and irregularity carry **negative** weights, meaning
the real arrangements are denser and less regular than the alternatives
generated on their own grids. This reflects the buildings in the corpus:
these are wall-bearing dwellings, in which columns stand where the load-
bearing walls stand, and those walls follow rooms rather than a regular
module. This result predicts the pattern of the Stage 12 accuracy tables.

## Files

- `ml_ranker.ipynb` — the formulation, the weights, and accuracy by
  data slice
- `train_learned_weights.py` — builds the comparisons, fits over five
  grouped splits, writes `data/learned_weights.json` and
  `data/rank_pairs_summary.csv`
- `figures/08_learned_weights.png`
