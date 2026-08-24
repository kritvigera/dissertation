# Stage 05: Arrangement generator

Lists the candidate column arrangements available on a parsed grid.

## What a candidate is

A candidate is a choice of grid lines together with a rule for which
crossings receive a column. The outermost line on each axis is always
retained; any subset of the interior lines is allowed provided every
resulting bay width lies within the user's minimum and maximum span.
Crossings excluded by constraint are removed, and such a removal is
permitted only if the neighbouring columns can still span the resulting
gap. Columns marked on the drawing are added to every candidate.

## Two placement rules

| Rule | Columns placed at | Available |
|---|---|---|
| `lattice` | every available crossing | always |
| `walls` | only crossings inside a load-bearing wall | when wall thicknesses are known |

The `walls` rule reproduces the procedure that generated the corpus
labels. Without it the correct arrangement would not be among the
candidates and the evaluation could not measure any variant against it.
Candidates that produce identical column sets under the two rules are
deduplicated.

## Sampling large search spaces

The number of line subsets grows exponentially with the number of interior
lines, so the space of candidate pairs cannot always be enumerated in full.
Where it exceeds the configured limit, the generator draws a random sample
of the pairs using a fixed seed, always including the pair that retains all
lines, which is where the correct arrangement lies. Sampling keeps the
candidate set representative of the whole space; simply stopping at the
limit would fill it with near-identical dense grids and silently discard
sparse ones. The same seed reproduces the same set.

At the default limit of 4,000 candidates, the seven held-out dwellings
produce between 512 and 4,000 candidates. Stage 12 reports the candidate
count for each plan alongside its accuracy figures. Sampling reduces
coverage without restoring exhaustiveness, and this is stated as a
limitation rather than presented as a solution.

## Files

- `arrangement_generator.ipynb` — the 33 candidates of the synthetic office
  plan, the candidates of a held-out dwelling under both placement rules,
  and the growth curve that makes sampling necessary
- implementation: `generate` and `valid_line_sets` in
  `09_full_pipeline/pipeline.py`
- `figures/05_example_arrangements.png`, `figures/05_search_space.png`
