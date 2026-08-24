# Stage 12, said simply

## What this stage does

Holds the exam. The four pipeline variants all sat the same seven tests —
seven real Swiss dwellings that no model ever saw during training — and
this stage marks the papers.

## Why an exam needs designing at all

Because without one, four pipelines produce four answers and nobody can
say which is better. The exam has to be fair in ways that are easy to get
wrong:

* **The right answer must be on the answer sheet.** Early versions of the
  pipeline could never actually propose the real building's arrangement —
  the parser read walls differently from the way the answer key was made,
  and the span rules excluded the real grid before scoring began. Every
  accuracy number would have measured that accident. It took real work
  (documented across stages 04 and 05) before the true arrangement
  appeared among the candidates on all seven plans, and the write-up
  reports that fact — the *reachability rate* — before any score.
* **One scoreboard, and an acknowledged consequence.** The question is
  whether the variant found the real arrangement, or something
  geometrically close to it. An earlier version also asked whether the
  chosen arrangement stood up well sideways, which was a check that did
  not depend on our own labelling. That check has been removed, so every
  number now rests on the labelling procedure, and the write-up says so.
* **Compare like a statistician, not a salesman.** Every comparison is
  *paired*: variant A and variant B on the same plan, differences across
  the seven plans, a signed-rank test. Seven plans is a small class, and
  the write-up says so plainly.

## What else gets tested here

All the "but what if you had chosen differently?" questions:

* shake each of the five score weights up and down by half — does the
  ranking care?
* turn the judge's influence (λ) from zero to loud — when does it matter?
* move the load-bearing wall threshold from 140 to 200 mm, since the
  entire answer key is built on that one number, so the exam is re-marked
  at each setting;
* check the judge is not just the score wearing a mask, by correlating
  their rankings.

## The one-line summary of the design

State the ceiling before quoting the score, pair every comparison, and
shake every dial that was set by hand.
