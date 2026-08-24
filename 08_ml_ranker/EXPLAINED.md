# Stage 08, said simply

## What this stage does

Stage 06 scores every layout with five measurements and five weights.
This stage asks twelve thousand real buildings what those weights should
be.

## How you can and cannot learn a weight

Two tempting shortcuts had to be refused.

You could train a model to copy the hand-set score — feed it layouts,
teach it to predict the penalty the equal weights give. It would learn
beautifully and prove nothing: you would be comparing the score with a
photocopy of itself. Refused.

You could reuse the judge's training signal — real layouts against
deliberately broken ones. But then the "learned score" and the "judge"
are the same animal wearing two hats, and comparing the pipeline
with-judge against without-judge stops meaning anything. Refused.

What is left is the honest version: **make it a contest**. For each real
building, line up the real arrangement against several alternatives built
on that same building's grid — the full every-crossing lattice, and
thinned-out variants. Ask: what five weights make the *real* one win as
often as possible? That question has a clean answer (a small logistic
model on score differences), and because the model is linear, its
coefficients simply *are* the five weights, readable in one row.

## What the buildings answered

With the learned weights, the real arrangement beats an alternative
**89 times out of 100** on plans the fit never saw; the equal weights
manage 75. And the learned row is worth staring at:

* *span demand* and *repetition* — positive, as designed. Real
  arrangements carry shorter loaded spans and repeat their bays.
* *off the walls* — the biggest weight by far. Real columns live in
  walls. (Partly a self-fulfilling truth here, since our ground truth was
  *derived* from walls — the notebook says this out loud.)
* *density* and *irregularity* — **negative**, upside down from the
  design intuition. Real derived arrangements are denser and messier than
  their alternatives. These are wall-bearing Swiss dwellings: columns
  stand where party walls stand, and party walls follow rooms, not neat
  6-metre modules.

That last bullet is the skeleton key for the results chapter: whenever
the equal-weight score ranks the true arrangement badly and the learned
score ranks it well, this is why.

## What used to be here

An earlier stage 08 was a "grid ranker" — a second learned model that
trimmed the shortlist from 15 to 5. It was retired on purpose: the
pipeline now has proper judges (stages 07, 11, 12), and two learned
opinions about the same geometry in one pipeline just count the same
evidence twice. Its old scores belong to its old task and are not quoted
against anything new.
