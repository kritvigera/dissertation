# Stage 03: Exploratory analysis

Describes the corpus produced by Stage 02, in order to establish how far
real floor plans differ from the regular grids an engineer would draw. That
difference determines what the judges in Stages 07 and 11 can learn, and it
explains why the learned score weights in Stage 08 disagree with
conventional design intuition.

Input: `data/features_all.csv`, 29,279 arrangements.

## Findings

**1. Column occupancy separates the two labelling procedures.** Under the
earlier procedure, which placed a column at every grid crossing, the column
count equalled the number of crossings in every plan by construction; the
arrangements were therefore artefacts of the parser. Under the derived
labels the median Swiss plan places columns at about 70% of its crossings
and the median CubiCasa plan at about 62%, while every one of the 16,812
ResPlan plans still shows exact equality. That equality identifies the
earlier procedure and is the reason ResPlan is excluded from training. A
derived feature, `fill_ratio`, previously carried this evidence; it has been
removed, being computable from three columns the table already holds, and
the counts are now reported directly.

**2. Building the grid from load-bearing walls only was the most valuable
change made to the pipeline.** Plans previously yielded a median of 84 to
130 candidate columns, where a structural grid for a dwelling is nearer 20
to 40, because every internal partition contributed a grid line. With the
grid built from load-bearing walls, the median falls to 40 for Swiss and 11
for CubiCasa. The proportion of plans with more than 100 columns falls from
71% to 9% for Swiss and from 33% to 0.1% for CubiCasa. ResPlan still shows
120 columns and 57%, because it provides no per-wall thickness and remains
on the earlier procedure.

**3. Real plans are irregular, which explains the failure of synthetic
training data.** Regularity, defined here as a measure that equals 1 when
every bay is the same size and falls towards 0 as bay sizes diverge, has a
median of 0.34 to 0.47 across the three sources. Published synthetic
training grids score far higher. The consequence was measured: a classifier
trained only on synthetic arrangements scored 0.57 ROC AUC on real ones,
where 0.5 is chance. Swiss is the most regular source at 0.468, consistent
with its being multi-storey apartment blocks, which makes it the closest
available proxy for the commercial buildings this pipeline targets.

**4. Irregularity is driven by the share of interior columns.** Setting
aside the two measures from which regularity is defined, the strongest
association is with the proportion of columns lying inside the building
rather than on its perimeter, at about −0.52. Plans whose columns are
mostly interior are markedly less regular, which is expected: interior
columns follow interior partitions, and partitions follow rooms. Larger
grids are also less regular, partly in their own right and partly as a
further consequence of finding 2.

## Consequences for later stages

1. Remove the 184 duplicate rows before training.
2. Never compare absolute measurements across sources without normalising.
3. Prefer Swiss Dwellings for structural conclusions; retain CubiCasa and
   ResPlan to test robustness across drawing conventions.
4. Expect the learned score weights to prefer denser and less regular
   arrangements than design intuition suggests, because that is what the
   corpus contains.

## Files

- `exploratory_analysis.ipynb` — the statistics and figures
- `figures/03_*.png`
- `data/summary_by_source.csv` — per-source medians
