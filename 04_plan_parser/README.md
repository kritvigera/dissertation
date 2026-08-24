# Stage 04: Plan parser

Converts wall segments into a structural grid and a set of candidate column
positions.

## Method

Each wall votes for a grid line. A wall running vertically votes for a
vertical line at its midpoint, weighted by its length; a horizontal wall
votes for a horizontal line in the same way. Votes closer together than a
merge tolerance are combined into a single line, positioned at the
length-weighted average of its members. Candidate column positions are the
crossings of the surviving lines.

Two rules govern which walls may vote. Walls shorter than a stated fraction
of the plan dimension are treated as drawing detail and are silent. Walls
more than 10 degrees from both axes are excluded entirely, because no
rotation straightens them and forcing them into the vote would place a grid
line where no wall exists; these are handled separately, by offering a
candidate position wherever a grid line crosses such a wall.

User constraints may exclude regions of the plan, marking the crossings
inside them as unavailable.

## Load-bearing walls

Where the input records a thickness for each wall, only walls at or above
the load-bearing threshold (160 mm by default) may vote, and every crossing
is tested for whether it lies physically inside such a wall, using that
wall's own thickness. This is the same test that produced the corpus labels
in Stage 02. Plans without thickness information, such as the synthetic
office plan and the DXF example, fall back to letting all walls vote and to
a fixed nominal width for the containment test.

This alignment is deliberate and it determines whether the evaluation is
meaningful. If the parser interpreted walls differently from the labelling
procedure, the correct arrangement could never appear among the candidates,
and every accuracy figure in Stage 12 would measure the disagreement
between two parsers rather than the quality of any variant. With the
alignment in place, the correct arrangement is reachable on all seven
held-out plans.

## Marked columns

Columns that the drawing itself marks are treated as input rather than as
part of the answer. The parser records them and Stage 05 retains them in
every candidate arrangement, matching both the labelling procedure of
Stage 02 and the treatment of "required" positions in the closest prior
work, US Patent 11,941,327 B2.

## Output

A grid description consumed by every later stage: the grid lines on both
axes, the crossings, which crossings are excluded by constraint, which lie
inside a load-bearing wall, the walls themselves with thicknesses, the
marked columns, the plan footprint and centre, and the angled-wall
crossings. The notebook writes the sample plan's grid to
`data/parsed_grid.json`.

## Files

- `plan_parser.ipynb` — the synthetic office plan, a held-out dwelling
  parsed with load-bearing awareness, the angled-wall case, and a check
  that the DXF reader reproduces the JSON result
- implementation: `parse_plan` in `09_full_pipeline/pipeline.py`
- `figures/04_parsed_plan.png`, `figures/04_parsed_dwelling.png`,
  `figures/04_inclined_walls.png`
