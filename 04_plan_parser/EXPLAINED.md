# Stage 04 explained in plain words

## What this stage does

Reads an architect's floor plan and works out where columns could go.

It does not decide where they should go. That comes later. This stage only
produces the list of places that are worth considering.

## The idea in one picture

An architect gives you a drawing full of walls. You want a structural grid. So
you draw straight lines right across the building wherever there is a
substantial wall, in both directions. Wherever two lines cross, a column could
stand. If the client has said "no columns in the atrium", you mark those
crossings as forbidden.

That is the whole stage.

## What goes in

Two files.

**The plan.** Either a `.json` file holding a list of walls, each wall being
four numbers, or a `.dxf` file, which is what CAD programs export.

**The constraints.** A small file where the user says what the project needs:

```json
{"n_stories": 6, "min_span": 6.0, "max_span": 12.0,
 "unit_scale": 1.0, "merge_tol": 0.6,
 "column_free_zones": [[15, 9, 21, 15]]}
```

The sample plan is drawn in metres, so those spans are 6 m and 12 m and the
atrium is the rectangle from 15 to 21 m across and 9 to 15 m up.

Nothing about the building is written into the code. Storeys, span limits, the
units the drawing uses, and any no column areas all come from this file. Change
the file, get a different answer.

`unit_scale` deserves a note. If a drawing is in millimetres and you want to
work in metres, set it to 0.001 and every number gets multiplied by that on the
way in. It is a small thing that prevents a very common and very expensive
category of mistake.

## Reading a DXF without a CAD library

A DXF file is not complicated once you look at it. It is a long list of pairs:
a number saying what kind of thing comes next, then the thing itself.

```
0
LINE          <- code 0 means "a new object starts", and it is a LINE
10
120.0         <- code 10 means "start x"
20
0.0           <- code 20 means "start y"
11
120.0         <- code 11 means "end x"
21
80.0          <- code 21 means "end y"
```

So the reader walks down the file in pairs. Every time it sees code 0 it
finishes whatever object it was collecting and starts a new one. If the object
was a LINE, it grabs codes 10, 20, 11 and 21 and writes down one wall. If it
was a polyline, which is a chain of points, it grabs all the points and turns
each neighbouring pair into a wall. Code 70 tells it whether the chain is
closed, and if so it joins the last point back to the first.

That is about 30 lines of Python and it means no CAD library is needed.

**What is not supported, and why it is written down rather than hidden.** PDFs.
A vector PDF has to be exported to DXF first, which every CAD program can do. A
scanned PDF, meaning a photograph of a paper drawing, would need image
recognition to find the walls, and that is a different project.

## Turning walls into grid lines

Every wall that runs along an axis votes for a line.

A wall running up and down votes for a vertical line at its x position. A wall
running left and right votes for a horizontal line at its y position.

### What about walls at an angle?

This is worth its own explanation, because the first version got it wrong.

The old code compared how far a wall travelled sideways against how far it
travelled up, and put it in whichever bucket was bigger. A wall at 30 degrees
would therefore be treated as horizontal, and a grid line would appear at its
average height, where no wall actually is.

Now a wall counts as orthogonal only if it is within 10 degrees of an axis.
Anything more slanted is called **skew** and is dealt with differently:

1. it does not vote for a grid line;
2. instead, every grid line is extended across the plan and the point where it
   **crosses** the inclined wall becomes a candidate column.

That crossing is a genuinely good place for a column. It is on a line the frame
already has, and it is on a wall that already carries load. Two things that
have to be true for a column to make sense.

Crossings that land exactly on an existing grid intersection are dropped,
because that candidate already exists.

### The knock-on effect

A column on an inclined wall usually sits part way along a bay rather than at a
corner. So the beam above it is no longer one long span, it is two shorter
ones. One honest note: in the held-out evaluation the inclined-wall
candidates are switched off, because the ground truth the variants are
measured against was derived on the orthogonal grid only and cannot
express a column off it. The path stays live in the app and in this
stage's notebook.

Then votes that are close together are merged. "Close" means within
`merge_tol`, which the user sets. Merging is weighted by wall length, so a 20 m
wall pulls the merged line towards itself much harder than a 2 m one.

The merging code is worth understanding because it is short and it looks
clever. Sort all the votes. Look at the gaps between neighbours. Wherever a gap
is bigger than the tolerance, cut. Every run between cuts becomes one line,
placed at the length weighted average of its members. Three lines of numpy, no
clustering library needed.

## Forbidden areas

A column free zone is written as a rectangle: `[x_min, y_min, x_max, y_max]`.
Any candidate column falling inside it is marked blocked. It stays in the list,
flagged, rather than being deleted, because stage 05 needs to know it was
removed on purpose.

## A worked example you can check by hand

There are two sample plans. The main one is a 6 storey office, 36 by 24
metres, with walls every 6 metres and an atrium in the middle that must stay
clear. `skew_plan.json` is the same building with a chamfered corner at 45
degrees and a diagonal spine wall at about 27 degrees, which is the one that
exercises the inclined-wall path.

Run the notebook and you get:

* 15 walls read
* 7 vertical grid lines, at 0, 6, 12, 18, 24, 30, 36
* 5 horizontal grid lines, at 0, 6, 12, 18, 24
* 7 times 5 = 35 candidate columns
* 1 of them blocked, the one inside the atrium

Every one of those numbers can be checked against the drawing with a ruler,
which is the point of using a small clean example first.

## Where this stage is weak

On real plans it produces far too many grid lines. Stage 03 measured it: 84 to
130 candidate columns per home, where a real structural grid would have 20 to
40.

Every partition wall votes, so a wardrobe gets the same say as a load bearing
wall. And a thick wall drawn with both faces produces two lines a few
centimetres apart.

The clean sample plan hides this completely, which is exactly why the honest
measurement was done on real data instead.

Three fixes in priority order: merge the two faces of a wall, rank lines by how
much wall length sits on them and drop the weakest, and refuse to create a line
that would make an unrealistically small bay.

## What comes out

`data/parsed_grid.json`, holding the grid lines, the walls, the blocked
positions and the constraints that produced them. Stage 05 reads it. Keeping
the constraints inside the file means later stages cannot accidentally use
different rules from the ones the grid was built with.
