# Stage 09, said simply

## What this stage does

Plugs the pieces together, four different ways, and gives you a web page
to play with them.

## The four variants

Every variant reads the plan the same way, lists the same candidate
layouts, and scores them with the same five-term penalty. They differ in
exactly one place — who gets the second opinion on the best fifteen:

* **V1** asks the feature judge (stage 07): six measurements, gradient
  boosting.
* **V2** asks the image judge (stage 12): a 64-by-64 picture, a small
  CNN.
* **V3** asks nobody. The score's favourite wins as it stands.
* **V4** asks the quantum judge (stage 11): four measurements, four
  qubits.

V3 is not a leftover — it is the control. The only way to know whether a
learned judge earns its keep is to run the same pipeline without one and
compare. Stage 12 does that comparison properly, with the same seven
held-out plans for everyone and statistics that respect the pairing.

The judge never overrules the score outright. Its opinion is blended in:
`penalty + λ × distrust`. At λ = 0 the judge is muted and every variant
becomes V3; crank λ up and the judge dominates. The default 0.3 is not
defended by argument — stage 12 simply tries a range of λ and reports
what changes.

## Two sets of weights

The five-term score comes in two flavours: weights all equal (the honest
default), or weights learned from twelve thousand real buildings
(stage 08). Every variant runs under both, which is how the thesis can
ask not just *which judge helps* but *which opinion about the terms is
right*.

## The web page

Start `app.py`, open localhost:8000, and drag things. Pick one of the
seven real held-out dwellings or the synthetic office; switch variants
and weight sets; move the span sliders and λ. The page draws the walls,
the grid, the chosen columns and the five score terms of the winner as
bars. Everything updates in about a second because the judges are trained
once when the server starts.

What you will not find anywhere on the page is a price — the pipeline no
longer computes one, for reasons stage 06 explains.
