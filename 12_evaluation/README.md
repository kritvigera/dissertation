# Stage 12: Evaluation

The measurement stage to which every other stage reports. Four variants,
sharing every earlier stage, are run end to end on seven held-out real
plans under both sets of score weights, and compared on their agreement
with the derived labels using paired statistical tests.

| Variant | Ranking path |
|---|---|
| V1 | score, then the feature judge |
| V2 | score, then the image judge |
| V3 | score alone — **the baseline the others must beat** |
| V4 | score, then the quantum judge |

Each judge re-orders the fifteen best-scoring candidates by combining the
normalised penalty with the judge's estimate of implausibility, weighted by
lambda. The penalty is normalised across the whole candidate set so that the
differences within the shortlist are small and the judge has real influence;
lambda is varied here rather than fixed by assertion.

## Candidate coverage is reported first

If the correct arrangement is absent from the candidate set, every accuracy
figure measures that absence rather than the quality of any variant. The
coverage rate is therefore reported before anything else.

Reaching **7 of 7** required aligning the parser in Stage 04 with the
labelling procedure of Stage 02, adding the wall-based placement rule to
Stage 05, treating marked columns as fixed input, widening the permitted
span range so that the correct arrangement's own grid lines are not
excluded in advance, and excluding angled-wall positions, which the
labelling procedure cannot represent. Each step is documented in the stage
it affects.

## What is measured

For every plan, variant and set of weights:

- whether the correct arrangement is the variant's first choice, and
  whether it is among its first five;
- the rank of the correct arrangement in the variant's full ordering, which
  is more informative than a hit rate when hit rates are low;
- two measures of near-agreement between the chosen columns and the correct
  ones: the average distance from each column to the nearest column in the
  other set, and the proportion of columns that coincide within a
  tolerance. Requiring an exact match would discard the distinction between
  a near miss and a complete failure.

## Statistical treatment

Variants are compared using paired tests across the seven plans, a Wilcoxon
signed-rank test together with an exact sign test, rather than by placing
two averages side by side. Seven plans provide limited statistical power,
which is stated with the results. The protocol scales without modification
to a larger held-out set.

Because agreement with the derived labels is now the only basis of
comparison, every figure this stage produces is conditional on the
labelling procedure of Stage 02. An earlier revision also measured a
lateral drift index, which was independent of those labels; it has been
removed, and the loss of that independent check is recorded in the
limitations of `THESIS.md`.

## Sensitivity analyses

| Analysis | File | Contents |
|---|---|---|
| Score weights | `evaluation_weights_sensitivity.csv` | each weight varied by ±50% and each measure removed in turn, with the resulting change in ranking order and in the top one and top five |
| Lambda | `evaluation_lambda.csv` | the judge's influence varied from 0 to 1 |
| Label threshold | `evaluation_threshold.csv` | the labels re-derived at 140, 160, 180 and 200 mm from the plans' own wall thicknesses, and the accuracy re-measured at each. Every figure in the thesis depends on this one constant, so its effect is measured rather than assumed |
| Score and judge overlap | `evaluation_overlap.csv` | the correlation between the score's ordering and the feature judge's across the whole candidate set. A high correlation would mean the judge contributes nothing and that variant V1 is effectively the baseline |

## Files

- `evaluate.py` — the protocol; use `--variants V1,V2,V3,V4` for the full
  run
- `evaluation.ipynb` — the tables and figures
- results: `data/evaluation_accuracy.csv`, `evaluation_tests.csv`, and the
  four sensitivity files above
