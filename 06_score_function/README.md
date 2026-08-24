# Stage 06: Score function

Assigns each candidate arrangement a single number, the penalty, computed
from the geometry of the arrangement and of the plan. Lower is better. The
penalty contains no member sizes, no load values, no prices and no units.

```
penalty = sum over measures of ( weight x normalised measure )
```

## Why the previous cost function was replaced

This stage formerly estimated a construction cost in pounds: it selected a
steel section for every beam and column, priced the steel by weight, added
a charge for each column standing away from a wall and for each additional
section size, and billed erection by the hour. Every one of those charges
rested on a rate that could not be sourced beyond "indicative figures". A
cost estimate is only as defensible as its rates, so the money was removed
and the structural reasoning behind it retained. The stage was renamed
accordingly.

## The five measures

Each is dimensionless, and each increases as the arrangement gets worse.

| Measure | Definition | Reasoning | Replaces |
|---|---|---|---|
| `t_density` | columns divided by floor area | Proxy for the quantity of vertical structure and the number of foundations, without pricing either | the material cost |
| `t_span_demand` | sum of (span squared x supported width), divided by a footprint term | The bending a beam must resist is proportional to the square of its span times the width of floor it carries. Removing the constant of proportionality leaves a quantity that still orders arrangements by structural demand, with no load value and no steel section required | the beam sizing |
| `t_irregularity` | variability of bay sizes in each direction, summed | Regular grids give predictable load paths and repeated details; grid regularity is the standard measure of early-stage layout quality | (new) |
| `t_off_wall` | fraction of columns not inside a wall | Columns away from walls intrude on the architecture and need non-standard detailing. This is the only measure that reads the architectural plan rather than the abstract grid, and it uses the same containment test as the labelling procedure, so training and use agree | the off-grid charge |
| `t_repetition` | number of distinct bay sizes divided by number of bays | Repeated bays mean repeated formwork and simpler procurement, which is a buildability argument that survives the loss of a section catalogue. Bay sizes within 5% of the mean count as the same | the section-variety charge |

`t_span_demand` is the direct replacement for the deleted beam sizing. The
structural relationship is preserved exactly, to within a constant factor
that is the same for every candidate; only the unsupported unit rates are
gone.

## Normalisation

Each measure is converted to a robust standard score within the candidate
set for one plan: the value minus the median, divided by the interquartile
range, with a fallback when that range is degenerate. Scaling between the
minimum and maximum was rejected because a single extreme candidate would
rescale every other score, making the result depend on how many candidates
were generated. The median and interquartile range are unaffected by values
outside the central half of the set.

## Two sets of weights

The weights are the function's only free parameters, and both sets are
carried through the evaluation.

**Fixed** (`SCORE_MODE="algo"`): all weights equal to one. Equal weighting
is the deliberate choice in the absence of an argued basis for preferring
one measure over another. Its justification is empirical rather than
rhetorical: Stage 12 perturbs each weight by ±50% and removes each measure
in turn, and reports the resulting change in the ranking. A weight vector
that cannot be argued for is acceptable precisely when the ranking can be
shown to be insensitive to it.

**Learned** (`SCORE_MODE="learned"`): fitted in Stage 08 by comparing real
arrangements against alternatives drawn from the same plan. The model is
linear in the differences between measures, so its coefficients are the
weights and can be read directly against the fixed set.

| | density | span demand | irregularity | off-wall | repetition |
|---|---|---|---|---|---|
| Fixed | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Learned | −0.42 | +0.38 | −0.46 | **+2.77** | +0.97 |

On held-out plans the learned weights rank the real arrangement above an
alternative in 89.4% ± 0.6% of paired comparisons, against 74.7% for the
fixed weights on the same data.

The signs are informative. Span demand and repetition behave as intended.
Off-wall dominates. Density and irregularity are **negative**, meaning the
corpus contains real arrangements that are denser and less regular than the
alternatives generated on their own grids. This follows from the buildings
themselves: in wall-bearing construction the columns stand where the walls
stand, and party walls follow rooms rather than a regular module. This
single result explains the pattern of the Stage 12 accuracy tables.

Two qualifications travel with it. The off-wall measure is zero for every
derived label by construction, since derived columns lie inside walls by
definition, so part of its large weight restates the labelling procedure
rather than discovering a property of practice. And both sets of weights are
evaluated under the identical Stage 12 protocol, so the comparison between
them is measured rather than assumed.

## Scope

This stage is not a cost model, a design check or a code compliance check.
Its only purpose is to order a candidate set so that a judge, or no judge
at all in the baseline variant, selects from the better end of it. The
span-demand measure is the only point at which structural reasoning enters
the pipeline, and it enters as a proportionality rather than as an
analysis. No stage of the pipeline computes structural behaviour.

## Files

- `score_function.ipynb` — the measures, the normalisation, the two weight
  sets, and a check on how far the measures duplicate one another
- implementation: `score_terms`, `score_arrangements`, `robust_z` and
  `ALGO_WEIGHTS` in `09_full_pipeline/pipeline.py`
- `figures/06_weights.png`
- weight sensitivity results: `data/evaluation_weights_sensitivity.csv`
