# The seven held-out plans

Seven real plans removed from the corpus and **excluded from every training
set**, so that no demonstration or measurement is performed on a plan any
model has already seen.

## Selection

The plans were chosen to be typical rather than favourable. Each plan in the
corpus was scored by its distance from the corpus median across seven
scale-independent measurements: column count, grid-line counts in each
direction, aspect ratio, regularity, interior-column proportion and the
ratio of mean bay sizes, expressed in standard deviations. The 120 closest
plans were then sorted by size and seven taken at even intervals, so that
the set spans small to large rather than repeating one size.

All seven come from Swiss Dwellings, which is drawn in metres and is the
only source where the 160 mm load-bearing rule applies as written. The plans
therefore carry real dimensions.

| Plan | Source id | Footprint (m) | Grid | Columns | Marked | Regularity | Distance |
|---|---|---|---|---|---|---|---|
| demo_01 | 5855 | 12.5 x 9.5 | 9 x 8 | 43 | 1 | 0.378 | 0.181 |
| demo_02 | 8491 | 15.5 x 13.6 | 10 x 9 | 72 | 0 | 0.406 | 0.109 |
| demo_03 | 44116 | 24.3 x 15.4 | 10 x 9 | 55 | 0 | 0.461 | 0.124 |
| demo_04 | 16422 | 19.8 x 16.5 | 9 x 8 | 42 | 8 | 0.399 | 0.125 |
| demo_05 | 2001 | 17.8 x 19.1 | 9 x 10 | 65 | 0 | 0.508 | 0.193 |
| demo_06 | 41319 | 24.4 x 17.3 | 10 x 8 | 68 | 0 | 0.417 | 0.160 |
| demo_07 | 840 | 36.3 x 37.7 | 10 x 11 | 56 | 0 | 0.423 | 0.184 |

"Marked" counts the columns the original drawing recorded explicitly, which
are retained wherever they fall. "Distance" is the typicality score above;
the 120th-closest plan in the corpus scores 0.199, so all seven fall inside
that set.

The set was reselected when the labelling procedure changed. Deriving
columns from load-bearing walls rather than from every wall crossing moved
the corpus distributions and therefore changed which plans count as typical;
only two of the previous seven remained within the 120 closest. The same
documented method was re-run rather than the claim being weakened.

## File contents

Each file holds the wall list in metres, a storey count and an empty list of
excluded regions, together with two fields added for the evaluation:

- **wall thickness**, as a fifth number on every wall, so that the 160 mm
  rule and its 140, 180 and 200 mm variants can be applied to these files
  directly without the original archive;
- **`annotated_columns`**, the coordinates of the columns the original
  drawing marked, which Stage 04 treats as fixed input that every candidate
  arrangement must retain.

The wall centre lines are taken directly from the Swiss Dwellings drawings,
rotated onto the axes in the same frame of reference used by the labelling
procedure and shifted to the origin, with `origin_shift` recording the
shift. Parsing them reproduces the plan's real grid because it is the plan's
real geometry.

The derived arrangement of each plan, at each threshold, is held in
`ground_truth.json` in this folder rather than in the plan files, so that
the pipeline cannot read its own answers. Stage 12 re-derives each label
from the plan's own walls and checks it against that file.

## Enforcement of the holdout

`data/holdout_ids.csv` lists the seven as source and identifier pairs. Every
training script removes them before constructing any dataset, through a
single shared loading function rather than in four separate places.

## What these plans are

These are dwellings, not offices. Swiss Dwellings is a residential corpus,
so the footprints are 12 to 37 m across rather than the 36 by 24 m of the
synthetic office plan. That reflects what the models were trained on, and it
is a reminder that the corpus is residential while the pipeline is aimed at
commercial frames.

It is equally important to be clear about what the columns in these plans
are. Swiss residential buildings are wall-bearing: the partition walls carry
the load and framed structures are the exception. The arrangement derived
for each plan is an equivalent-frame idealisation of that wall system, not a
layout an architect drew. Only the "Marked" column counts positions the
drawing actually recorded. See `02_data_pipeline/README.md` for the full
statement.
