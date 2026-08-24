# Stage 05 explained in plain words

## What this stage does

Takes the candidate column positions from stage 04 and makes a list of every
sensible column layout you could build from them.

Not the best one. All of them.

## Why "all of them" is a reasonable thing to do

Here is the trick that makes this possible.

The grid lines at the very edge of the building are not optional. You need
columns at the corners and along the edges, so the first and last line in each
direction always stay.

That leaves only the interior lines, and for each of those there is one yes or
no question: keep it or drop it?

So a layout is nothing more than a choice of which interior lines to keep. With
5 interior lines there are 2 to the power 5, which is 32 possible answers. With
8 there are 256. These are small numbers for a computer, so instead of
searching cleverly, the code just tries every single one.

The advantage is not speed. It is certainty. A clever search can miss the best
answer and you would never know. Checking everything cannot.

## The rules a layout has to pass

The user gives a minimum and maximum span. In the sample they are 6 and 12
metres.

* **Minimum span.** Columns too close together waste money and get in the way.
* **Maximum span.** Beams that span too far get deep and heavy, or simply do
  not work.

Any combination of lines where some bay falls outside that range is thrown away
immediately. Not built and then filtered. Never built at all. That is why the
code has no "check if valid" step at the end.

## The awkward case: a removed column

Suppose the atrium forces you to drop one column that would otherwise have sat
at a crossing you kept.

The beams either side of that column now have to span the whole way across the
hole. So the effective span there is not one bay, it is two bays joined
together.

The code checks this specifically. For each blocked position it measures the
distance from the previous kept line to the next kept line, in both directions,
and rejects the layout if that doubled distance breaks the maximum span rule.
Missing this check would let the program propose layouts with beams that cannot
be built.

## What actually happens on the sample plan

```
7 vertical lines  ->  13 valid combinations
5 horizontal lines ->   5 valid combinations
13 times 5        ->  65 possible pairs
after the blocked column check and the minimum of 4 columns:  33 layouts
```

So 33 real, buildable options come out of one plan, every one of them
respecting the user's span limits. On plans that carry wall thicknesses —
the seven held-out dwellings do — each set of lines yields up to two
candidates rather than one: the full lattice, and a "walls" version with
columns only where a load-bearing wall actually passes through the
crossing. That second kind is how the real, derived arrangement gets to
be one of the candidates instead of an unreachable ideal.

## The honest problem

This only works because the sample plan has few interior lines.

Real plans, after the over-segmentation described in stage 03, have around 40
interior lines per direction. Two to the power 40 is about a thousand billion.
Checking every option stops being clever and becomes impossible.

So there are two ways forward, and they are worth stating plainly:

1. Fix the parser so real plans produce 20 to 40 candidates instead of 130. Then
   full enumeration keeps working and keeps its guarantee.
2. Give up the guarantee and search cleverly instead.

What the generator does today is a middle path: when the space overflows
its cap it takes a *seeded random sample* of it — always including the
full-line-set option — instead of stopping at the first few thousand it
happens to meet. Sampling keeps the candidate set representative; it does
not restore the guarantee, and the write-up says which plans hit the cap.

## Why not use a generative model instead

There is a large body of work on using deep learning to generate structure,
usually by producing an image where each pixel says how much material should be
there.

Those methods scale beautifully and produce beautiful pictures. The problem is
what you get at the end: a cloud of material density, not a set of column
positions with spacings a contractor can set out on site. Turning that cloud
into a buildable grid is the hard part, and it is the part those methods leave
out.

Enumerating discrete layouts means every candidate is buildable before anyone
looks at it. That is worth a lot more than elegance.

## What comes out

`data/arrangements.json`, a list of 33 layouts, each with its kept grid lines,
its column positions, and the storey count. Stages 06 onwards all read it.
