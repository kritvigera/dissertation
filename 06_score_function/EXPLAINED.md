# Stage 06, said simply

## What this stage does

Takes every candidate column layout and gives it one number: a penalty.
Small penalty, good layout. The four pipeline variants all start from this
ordering; three of them then let a learned judge adjust the top of it.

## What happened to the prices

This stage used to answer in pounds. It picked a steel beam for every span
from a catalogue, a column section for every load, priced the tonnage,
charged £350 for a column off the wall grid, and billed the erection crew
by the hour. All of it is gone.

Why? Because every one of those pounds rested on a rate someone had to
invent. £350 per awkward column — measured where? £90 an hour — which
crew, which year, which country? In an examination the first question about
a cost model is where the rates came from, and "indicative figures" is not
an answer that survives. The safe version of this stage keeps the
*reasons* behind those charges and drops the currency.

## The five things it now measures

1. **Density.** How many columns for how much floor. More columns means
   more members and more foundations. The old version priced this in
   steel; now it is just a count over an area.
2. **Span demand.** For every stretch of floor between two columns, the
   square of the span times the width of floor it carries. This is the old
   beam-sizing physics with the price stripped off: an engineer sizes a
   beam from w·L²/8, and if you drop the w/8 — the same for every
   candidate — what remains still says *this layout bends harder than that
   one*. No load value, no catalogue, same ordering logic.
3. **Irregularity.** How uneven the bay widths are. Even bays mean
   predictable load paths and details that repeat.
4. **Off the walls.** What share of the columns land outside the
   building's walls. A column in the middle of a room upsets the architect
   and needs its own transfer details. This is the one term that looks at
   the drawing rather than the grid.
5. **Repetition.** How many *different* bay sizes the layout uses. Two
   sizes means one set of formwork used many times; nine sizes means nine.

Each is a pure number, each gets standardised against the other candidates
of the same plan, and the standardising uses the median and the middle
half of the data — not the minimum and maximum, which one freak candidate
can drag anywhere.

## Who sets the five weights

Two answers, both shipped, both tested:

* **All equal.** Nobody can prove density matters exactly as much as
  repetition — so nobody pretends to. Instead, stage 12 shakes every
  weight up and down by half and shows how little the ranking moves. A
  number you cannot justify is fine if the answer barely depends on it.
* **Learned from the buildings.** Stage 08 fits the weights so that, for
  each real plan, the real arrangement scores better than the alternatives
  on its own grid. The learned weights disagree with intuition in a useful
  way: real arrangements turn out to be *denser* and *less tidy* than the
  alternatives — because these are wall-bearing Swiss buildings, and the
  columns stand where the walls stand, not on a neat module. Keep that in
  mind whenever a result later on looks surprising; it usually traces back
  to this.

## What to remember

One number per layout, built from five reasons an engineer would
recognise, with no invented prices — and the weights either defended by a
sensitivity test or learned from twelve thousand real buildings.
