# Stage 03 explained in plain words

## What this stage does

Looks at the 29,279 layouts from stage 02 and asks: what are these actually
like? Before building any model, you look at your data. Skipping this is how
people end up training models on rubbish.

## The one question behind all of it

Engineers draw tidy grids. Columns in neat rows, bays all about the same size.
Architects draw rooms, and walls go wherever rooms need them.

I am about to learn from architects' drawings and use it for engineering. So:
how far apart are those two things? The answer decides how much any of the
later stages can be trusted.

## Words you will meet

* **Median.** The middle value. Line up all the numbers and take the one in the
  middle. Better than an average when a few extreme values would drag the
  average around.
* **Coefficient of variation.** How spread out numbers are, compared to their
  own size. Bays of 5, 6 and 7 metres are fairly even. Bays of 1, 6 and 20 are
  not. This number captures that in one figure.
* **Regularity.** My own measure, built from the above: `1 / (1 + cov_x +
  cov_y)`. It runs from near 0 to 1. A value of 1 means every bay is the same
  size. Real plans come out between 0.34 and 0.47.
* **Correlation.** Whether two numbers move together. Runs from -1 to +1. Zero
  means no relationship.
* **Constant feature.** A number that is the same for every single row. It
  tells you nothing at all, and it should be deleted.

## Finding 1: the occupancy story, told by the counts

How many of a plan's grid crossings actually carry a column? Under the old
parser the answer was *all of them, always* — a column at every crossing by
design, in every single row. Those were fabricated lattices, not
arrangements, and a feature computed from them (`fill_ratio`) sat at
exactly 1.0 and told you nothing.

Two things have happened since. Stage 02 now derives columns from
load-bearing walls, so real occupancy varies: the median Swiss plan uses
about 70 percent of its crossings, CubiCasa about 62, while ResPlan still
reads exactly 100 percent in every row — the fingerprint of the old path,
and the reason ResPlan is excluded from training. And the `fill_ratio`
column itself has been dropped from the feature table: it was one column
divided by two others that the table already carries, and a ratio a model
can rebuild for itself is bookkeeping, not information.

The lesson survives the feature: a number that never changes tells you
nothing, and the notebook still checks for constant features
automatically so the same mistake cannot happen twice.

Also found: 184 rows share an identical set of feature values, about 0.6
percent. They are genuinely different layouts that happen to have the same
shape description. Not a bug, but worth removing before training.

## Finding 2: the naive parse produced far too many columns — and the fix worked

This was the big one, and fixing it changed every later stage.

Under the original every-wall parse, the middle plan carried 120 columns in
ResPlan, 130 in Swiss and 84 in CubiCasa — where a real home's structural
grid has something like 20 to 40. Every partition voted for a grid line: a
wardrobe wall got the same say as a load-bearing wall, and a thick wall
drawn with two faces produced two lines 200 mm apart. Worst of all, the
damage sat in the training data itself, so every learned stage inherited a
distorted picture of what buildings look like.

The fix was the load-bearing derivation of stage 02: keep only walls at
least 160 mm thick, grid from those, and place columns only where such a
wall actually passes through a crossing. The medians fell to 40 for Swiss
and 11 for CubiCasa — into the believable range — and the share of Swiss
plans with over 100 columns collapsed from 71 percent to under a tenth of
that. ResPlan publishes no per-wall thickness, so it cannot take the fix:
its median stays at 120, which is why its rows are excluded from training
and kept only as a cautionary exhibit of the naive path.

## Finding 3: real plans are far less regular than the synthetic ones

Median regularity is 0.34 to 0.47. Synthetic training sets, of the sort used in
published work, have uniform bays of 28 to 40 feet and score much higher.

That gap is not just a curiosity. It was measured directly: a model trained
only on synthetic layouts and then tested on real ones scored 0.57 on a scale
where 0.5 is random guessing. Practically speaking, it had learned nothing that
transferred.

That single number is why stage 08 was rewritten to train on this corpus
instead of a borrowed synthetic one.

One more useful detail. Swiss is the most regular source at 0.468. That fits:
Swiss plans are apartment blocks, which are more disciplined than one off
houses. So Swiss is the closest thing in this data to the commercial buildings
this pipeline is aimed at.

## Finding 4: what makes a plan irregular

Correlations with regularity:

| Feature | Correlation | What it means |
|---|---|---|
| `cov_x`, `cov_y` | -0.91 | Ignore this one, regularity is built from them |
| `interior_ratio` | -0.52 | More interior columns, less regular |
| `n_columns` | -0.39 | Bigger grids, less regular |
| `aspect_ratio` | +0.26 | Long thin plans slightly more regular |

The interesting one is `interior_ratio`. Plans where most columns are inside
the building rather than round the edge are noticeably less regular. That makes
sense: interior columns come from interior partitions, and partitions go
wherever the rooms want.

Part of the "bigger grids are less regular" effect is real, and part of it is
just Finding 2 showing up again.

## What this stage changed downstream

1. `fill_ratio` deleted.
2. Absolute size features never compared across collections. This is why stage
   07 uses only six shape features.
3. Fix the parser before collecting more data, since more distorted plans do
   not help.
4. Swiss preferred for anything structural, the other two kept because they
   force the code to handle different drawing habits.

## Check it yourself

Run the notebook. It reads the committed table, so it works immediately. Every
number quoted above is printed or plotted by it, and the quality checks at the
top run automatically, including the constant feature check that caught
`fill_ratio`.
