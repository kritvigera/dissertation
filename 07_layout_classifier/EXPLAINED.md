# Stage 07, said simply

## What this stage does

Trains a judge. Show it a column layout, it answers with a probability:
*does this look like the arrangement of a real building?* In pipeline
variant V1, that judge gets the fifteen best-scored layouts and nudges the
ordering — a layout the score likes but the judge distrusts slips down.

## The mistake this version fixes

A judge learns from examples: real layouts labelled *good*, fakes
labelled *bad*. The judge is only as honest as the fakes.

The old fakes were made by editing the *measurements* of a real layout —
take its row of six numbers, multiply the "unevenness" number by four,
hand the row back. The problem: the judge reads those same six numbers.
It never saw a building; it saw which dials had been turned, and it
learned to spot turned dials. One dial alone gave most of the answer
away. The score looked excellent and meant almost nothing.

The new fakes are made by moving the *columns*. Four ways:

* **transplant** — steal another real building's layout and stretch it
  onto this plan's outline. Everything about it is genuinely
  building-like; it is just the *wrong building's* answer;
* **jitter** — shake every column off its line;
* **shear** — tilt the whole arrangement so nothing is square;
* **dropout** — delete a third of the columns and shove some survivors.

Only *after* the columns have moved are the six measurements taken. The
judge now has to tell real geometry from broken geometry, which was the
assignment all along.

## What the honest numbers look like

The old headline was 0.947 (out of 1). The new one is **0.862** — and the
drop is the finding. About a tenth of the old score was the judge
grading its own fake-maker. Three more numbers worth knowing:

* the best a *single* measurement can do is now 0.679 — no shortcut left;
* transplants fool the judge most (0.668): they are real layouts, just
  borrowed, and that is the hardest and fairest test;
* a judge trained only on Swiss buildings scores 0.673 on Finnish ones
  (CubiCasa), while the reverse trip scores 0.805 — models do absorb one
  country's drawing habits, and the thesis says so instead of pooling
  quietly.

## Where it sits in the bigger comparison

The exact same real examples and the exact same fakes are also shown to
the picture-judge (stage 12) and the quantum judge (stage 11). Same
question, same answer key, three very different students — that
controlled comparison is the thesis's main event, and stage 12 is the
exam hall.
