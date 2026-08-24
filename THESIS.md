# Structural arrangement prediction from architectural plans

**A controlled four-way comparison of an algorithmic score, a feature-based
judge, an appearance-based judge and a quantum judge, on one task, one
corpus and one set of labels.**

Master's thesis, Imperial College London. This document is the whole
project in one place: the problem, the data, the methods, the models, the
results and their limits. Each numbered section names the stage folder
that holds the code and the longer write-up.

## Contents

1. Introduction — the problem, the four variants, what is new
2. Literature — the closest prior work and the gap
3. Method — from raw archives to the evaluation protocol
4. The models that were trained
5. Results
6. Discussion
7. Limitations
8. Conclusions and further work
9. How to reproduce all of it

---

## 1. Introduction

### 1.1 The problem

Early in a building's design, someone has to propose where the columns go.
The architectural plan exists; the structural grid does not. The decision
is made quickly, from experience, and it constrains everything that
follows — spans, floor depths, foundations, and how much of the
architect's plan survives contact with the structure. This thesis asks
whether that first proposal can be generated, ranked and measured
automatically, learning what a plausible arrangement looks like from real
buildings rather than from hand-written rules.

### 1.2 What the thesis asks

Three questions, in rising order of importance:

1. Can candidate column arrangements be enumerated from a plan so that the
   real building's arrangement is *among* them? (If not, nothing
   downstream can be measured honestly.)
2. Can a purely geometric score — no loads, no sections, no prices — order
   those candidates so the real arrangement sits near the top?
3. Do learned judges — a feature model, an image model, a quantum model —
   improve on the score alone, when all three are trained on one identical
   task and the score-only pipeline is kept as the baseline?

### 1.3 What was built

```
plan + constraints
  [04] parse      walls -> grid lines -> candidate columns (load-bearing aware)
  [05] enumerate  every arrangement within the span limits, two placement
                  policies, seeded sampling past the cap
  [06] score      five dimensionless terms -> penalty; weights equal ("algo")
                  or learned by pairwise ranking ("learned", stage 08)
   +- V1  [07] ML1, gradient boosting on six scale-invariant features
   +- V2  [12] ML2, a small CNN on 64x64 rasterisations
   +- V3  no judge: the score's order stands        <- the baseline
   +- V4  [11] a 4-qubit variational quantum classifier
  best 1
  [12] evaluate   reachability first, then the 4x2 accuracy table, paired
                  tests, and every sensitivity analysis
```

The judges re-order the score's top 15 through
`combined = z(penalty) + λ·(1 − P(plausible))`, with the penalty
normalised over the full candidate set and λ ablated. There is no pricing,
no member sizing and no load value anywhere in the ranking path; the root
README's changelog records what was deleted and where each piece of
structural reasoning went.

### 1.4 Objectives

1. Derive a column-arrangement ground truth for open floor-plan corpora,
   document every assumption, and measure the sensitivity of the results
   to the one arbitrary constant in it.
2. Align candidate generation with that derivation until the ground truth
   is reachable, and report the reachability rate before any accuracy.
3. Build a score function whose only free parameters are five readable
   weights, and obtain those weights two independent ways.
4. Train three judges — features, pictures, qubits — on one identical
   task with shared geometry-space negatives.
5. Compare all four pipeline variants under one protocol, on agreement
   with the derived ground truth, with paired statistics.
6. Leave every known loophole either closed or stated.

### 1.5 What is new here

Checked against the 2020–2026 literature in August 2026; Section 1.9 of
the review records the evidence, and Section 1.10 the closest prior work,
including those that weaken the claims. Stated narrowly:

1. **The controlled four-way comparison** — algorithmic score, feature
   judge, image judge, quantum judge, on one identical task, corpus and
   label set, with the score-only baseline retained (V3), matched-capacity
   quantum controls, and paired tests. Every *pair* of judge families
   exists somewhere; no found study runs even three on one design task.
2. **A derived, sensitivity-swept column ground truth for open corpora** —
   the label problem (drawings record no columns) solved in public, with
   coordinates committed and the 160 mm constant swept 140–200 mm.
3. **Score weights learned by ranking** real derived arrangements against
   alternatives enumerated over the same plan's grid — the only
   formulation that is neither a tautology on the algorithmic score nor a
   reuse of the judges' signal.
4. **The feature-versus-appearance judging comparison** on identical rows
   and labels, with the wall-channel ablation that locates the image
   judge's advantage in the plan context rather than in convolution.
5. **A quantum judge of design-stage arrangement plausibility** — to the
   best of a systematic search, the first VQC applied to structural
   layout plausibility (post-earthquake safety classification exists and
   is cited), framed strictly as feasibility.
6. **An evaluation protocol that reports its own ceiling** — reachability
   before accuracy, geometry-space negatives, grouped and repeated
   splits, PR-AUC beside ROC AUC, leave-one-source-out transfer.

### 1.6 What is not claimed

The walls-to-gridlines-to-intersections decomposition (Autodesk patent
US 11,941,327 B2). CNN column-layout generation from plans (Ampanavos et
al. 2021, synthetic frames). Learning structural layout from real
buildings (Liao et al. 2021; Pizarro et al. 2021). Quantum classification
of buildings (Bhatta & Dang 2024). Quantum advantage of any kind. And no
individual model here is state of the art at anything; the contribution
is the comparison and the measurement discipline around it.

---

## 2. Literature

The full review is `01_literature_review/README.md`: 43 verified sources,
a decision register mapping each design choice to its evidence, the
novelty assessment (Section 1.9) and the closest prior work (Section 1.10).
The four works
an examiner would reach for first:

- **US 11,941,327 B2** (Wang & Nourbakhsh, Autodesk): the same
  decomposition, as a patent — RL selection, undisclosed data, no
  learned-versus-rule comparison. This thesis contributes the open corpus,
  the derived labels, and the four-way ranked comparison; it concedes the
  decomposition.
- **Ampanavos, Nourbakhsh & Cheng 2021**: a CNN generating column layouts
  from plan sketches, trained on ~137k synthetic frames. Real-building
  labels and judging/ranking are what it lacks.
- **Pizarro et al. 2021**: shear-wall layouts predicted from plans,
  trained on 165 real (private) Chilean projects — the nearest
  real-buildings precedent; generative, no candidate enumeration.
- **Bhatta & Dang 2024**: a VQC classifying buildings safe/unsafe after
  earthquakes against ten classical models — the closest quantum work,
  and the reason claim 5 above says "design-stage arrangement
  plausibility" rather than "QML on buildings".

The gap, stated so as to survive scrutiny (review Section 1.11): open architectural
collections as structural training data; the column-label problem solved
in public with its sensitivity measured; and candidate rankers actually
compared against each other rather than existing side by side.

---

## 3. Method

### 3.1 The corpus and the derived ground truth (stage 02)

Three open floor-plan corpora are parsed into one format: Swiss Dwellings
(8,262 plans kept, metres, walls as WKT polygons), CubiCasa5K (4,205,
SVG pixels), ResPlan (16,812, no per-wall thickness — descriptive only,
excluded from all training). Total 29,279 arrangements; 12,467 with
derived ground truth.

Architectural drawings record no column layout, so one is **derived**:
discard walls thinner than 160 mm (the 2-hour fire-rating minimum,
bracketed by Eurocode 2's 140 mm floor and DISCS's 200 mm median);
build the grid from the survivors; place a column at every grid node
physically inside such a wall; union with the columns the drawing
annotates. Steps 1–3 are an equivalent-frame idealisation of wall-bearing
buildings — Swiss residential *is* wall-bearing (Badoux & Peter) — and
only step 4 is observed. The derivation is never read as the building's
real structural system, and every number in this thesis is conditioned on
it; §5.6 measures by how much.

New in this revision: the regeneration (`regenerate_corpus.py`) validates
itself against the previous corpus (Swiss matched on all 8,260 shared
ids; CubiCasa on all 4,205) and persists what earlier runs discarded —
the ground-truth **column coordinates** and per-plan **load-bearing
walls**, which is what makes the judges, the learned weights and the
evaluation trainable from a clone; plus the 140/160/180/200 mm sweep, and
the seven demo plans re-exported with per-wall thickness and a separate
answer-key file.

Corpus-level threshold sensitivity: the median Swiss plan carries 52
derived columns at 140 mm, 40 at 160, 30 at 180, 24 at 200. The label is
not fragile to ±20 mm and is strongly conditioned beyond that; §5.6
carries this into the headline accuracy.

### 3.2 What the data looks like (stage 03)

Median regularity 0.34–0.47 across sources — far below the tidy synthetic
grids of published training sets. The naive every-wall parse produced
84–130 columns per plan; the load-bearing derivation brings Swiss to 40
and CubiCasa to 11, while ResPlan (no thickness available) stays at 120
as the cautionary exhibit. Absolute features are never compared across
sources; everything cross-source is scale-invariant or rasterised.

### 3.3 Parsing and enumeration (stages 04, 05)

The parser votes walls into grid lines exactly the way the corpus
derivation does — load-bearing walls only where thickness is known,
midpoint votes weighted by axis-length, a 1.5 % merge tolerance, short
walls silent — because any divergence there caps every downstream
accuracy number (loophole L4). Drawn columns are fixed input every
candidate keeps. The generator enumerates line subsets within the span
limits under two placement policies (full lattice; columns only inside
load-bearing walls) and **samples** the pair space with a fixed seed when
it overruns the cap, always keeping the full line set. Inclined-wall
crossings remain a live feature of the app but are disabled in the
evaluation, because the ground truth cannot express them.

### 3.4 The score function (stage 06)

Five dimensionless, monotone terms — density, span demand
(span² × tributary width: w·L²/8 with the constant dropped, the honest
residue of the deleted beam sizing), irregularity (cov of bays), off-wall
fraction (the derivation's own containment test), repetition (distinct
bay sizes) — each robustly normalised (median/IQR; min–max rejected,
loophole L5) within the candidate set, then combined with weights that
are the function's only free parameters. `algo` mode uses ones and is
defended by the sensitivity analysis; `learned` mode uses stage 08's
vector.

### 3.5 The learned weights (stage 08)

Learning to rank: the derived ground truth versus up to five alternatives
enumerated over the same plan's grid (52,724 pairs from 11,098 plans), a
logistic model without intercept on differences of the normalised terms,
five grouped splits. Two formulations were rejected as unsound —
regressing onto the algorithmic score (tautology, L6) and reusing the
judges' real-versus-corrupted signal (which would collapse V1 into V3).
The old grid ranker that lived in stage 08 is retired for the same
double-counting reason (L7).

### 3.6 The judges (stages 07, 12, 11)

One dataset feeds all three: positives are the 12,460 usable derived
arrangements (demo plans excluded), negatives are **geometry-space**
corruptions of the column coordinates — cross-plan transplant, jitter,
shear, dropout — generated before any feature or pixel exists. This
replaces the earlier feature-space corruptions, which perturbed three of
the six features the model then read (loophole L3, now closed rather
than documented).

- **ML1** (V1): gradient boosting on six scale-invariant features, the
  restriction forced by the corpus's three unit systems.
- **ML2** (V2): a ~15k-parameter CNN on 64×64 two-channel rasterisations
  (load-bearing walls; columns), bounding box fitted to the canvas so
  units vanish — the design trade stated up front: ML2 cannot know a span
  is 6 m rather than 6 ft (L8). Negatives are rasterised after
  corruption, never perturbed in pixel space.
- **The quantum judge** (V4): a 4-qubit variational classifier (angle
  embedding, strongly entangling layers; plus a data re-uploading variant)
  on the top four features, `default.qubit` simulator. Stage 10 fits it at
  400, 2,000, 8,000 and ~19,900 training rows against **one fixed test
  set**, with a capacity-matched logistic control on the same features and
  rows and full-capacity boosting controls (L9). The judge the pipeline
  uses trains on the canonical 80 % training partition
  (`data/train_test_split.csv`, grouped by plan, drawn after the seven
  demonstration plans are removed), so all three judges see identical
  data and none has seen the 20 % test partition or the demonstrations.

### 3.7 What the pipeline does not compute

No stage of the pipeline performs structural analysis. Two earlier
revisions did: one checked every member against the Eurocodes, using the
sections the deleted cost function had selected, and a later one computed
a position-sensitive lateral drift index that served as a second,
independent basis of comparison. The member check could not outlive member
sizing, and the drift index has since been removed as well.

The consequence is stated rather than minimised. Structural reasoning now
enters the thesis at exactly one point, the span-demand measure of
Section 3.4, and it enters as a proportionality rather than as an
analysis. Agreement with the derived labels is therefore the sole basis of
evaluation, and every figure in Section 5 is conditional on the labelling
procedure of Section 3.1. Section 7 records this as a limitation, and
Section 8 names the reinstatement of a label-independent check as further
work.

### 3.8 The evaluation protocol (stage 12)

Seven real Swiss dwellings, held out of every training set, each with
walls, thicknesses and drawn columns committed. Per plan: parse,
enumerate (512 to 4,000 candidates), re-derive the ground truth from the
plan's own walls (validated against the corpus answer key), then run
each variant under both score modes. Metrics: top-1, top-5, rank of the
ground truth, Chamfer distance and tolerance-matched IoU of the chosen
columns. Comparisons are paired (Wilcoxon
signed-rank plus sign test) across plans. Sensitivity: score weights
±50 % and leave-one-term-out (Kendall's τ), λ over {0…1}, the ground
truth re-derived at 140/160/180/200 mm, and the score-judge rank overlap
(L7).

---

## 4. The models that were trained

| Model | Data | Split | Reported |
|---|---|---|---|
| ML1 feature judge (07) | 24,920 rows: 12,460 positives + geometry-space negatives | grouped by plan, ×5 seeds | ROC AUC, PR AUC, per source, LOSO, stump baseline, per-kind |
| ML2 image judge (12) | the same rows, rasterised | grouped by plan, ×3 seeds | the same, plus the paired feature control and the wall-channel ablation |
| Learned weights (08) | 52,724 ranking pairs, 11,098 plans | grouped by plan, ×5 seeds | pairwise accuracy vs the algorithmic vector, coefficients ± std |
| Quantum judge (10) | 400 to 19,936 training rows, 4 selected features, one fixed test set | grouped by plan, single split per size (stated) | ROC AUC and PR AUC at each size, against capacity-matched and full-capacity controls |

The seven demo plans are excluded from every one of these through a
single shared loader. Class balance is synthetic 1:1, so PR-AUC is
reported beside ROC AUC and neither is a deployment claim (L13).

---

## 5. Results

### 5.1 Reachability, before anything else

**7 of 7** held-out plans contain the derived ground truth organically in
their candidate set. This did not happen by luck: it required the parser
alignment, the walls placement policy, fixed drawn columns, a span band
of [0.1, 10] m (the ground truth's own lines can sit 0.13 m apart — a
consequence of deriving labels from wall geometry, stated rather than
hidden), and disabling skew crossings. Accuracy below is therefore
unconditional; no ceiling correction is needed.

### 5.2 The judges on their own task

| | ROC AUC | PR AUC |
|---|---|---|
| ML1, grouped ×5 | 0.862 ± 0.006 | 0.831 ± 0.008 |
| best single-feature stump (regularity) | 0.679 ± 0.008 | — |
| ML2 CNN, walls + columns | **0.995 ± 0.003** | 0.995 ± 0.003 |
| ML2 CNN, columns only | 0.874 ± 0.019 | 0.819 ± 0.032 |
| ML1 paired control on ML2's splits | 0.860 ± 0.007 | 0.830 ± 0.010 |

Leave-one-source-out: ML1 travels asymmetrically (Swiss→CubiCasa 0.673,
CubiCasa→Swiss 0.805); the two-channel ML2 barely notices (0.983 /
0.988). Against negatives by kind, ML1 finds transplants hardest (0.668)
— real arrangements, merely borrowed — and jitter/shear easy
(0.956–0.960).

Three sentences carry this table. **The honest ML1 headline is 0.862,
down from 0.947** — the difference was the model reading its own
feature-space negative generator, and the drop is reported as the
finding it is. **Pictures and features tie at equal information**
(0.874 vs 0.860, columns only). **The picture wins outright when it can
see the walls** (0.995) — and that is the plan context that
scale-invariant features cannot carry, not convolution reading column
patterns better. The caveat attaches here as everywhere this signal
appears: column-wall registration is partly the derivation restated.

### 5.3 The weights: learned versus algorithmic

Pairwise accuracy on held-out plans: **0.894 ± 0.006** learned versus
**0.747** for the equal vector on the same folds. The learned vector:

| | density | span demand | irregularity | off-wall | repetition |
|---|---|---|---|---|---|
| learned | **−0.42** | +0.38 | **−0.46** | +2.77 | +0.97 |

Span demand and repetition agree with the design intuition; off-wall
dominates (partly by construction — the ground truth's off-wall fraction
is zero by definition); and density and irregularity come out
**negative**: the derived real arrangement is denser and less regular
than enumerated alternatives to its own grid, because wall-bearing
buildings put columns where the party walls are. This one sign flip
predicts the whole shape of §5.5.

### 5.4 The quantum comparison

ROC AUC on one fixed held-out test set, by training-set size:

| model | 400 | 2,000 | 8,000 | 19,936 | gain |
|---|---|---|---|---|---|
| boosting, 6 features | 0.790 | 0.841 | 0.860 | **0.865** | **+0.075** |
| boosting, same 4 features | 0.705 | 0.753 | 0.779 | 0.789 | **+0.084** |
| logistic, same 4 (capacity-matched) | 0.747 | 0.749 | 0.746 | 0.746 | −0.001 |
| VQC, 2 layers | 0.741 | 0.751 | 0.734 | 0.748 | +0.007 |
| data re-uploading, 6 layers | 0.741 | 0.761 | 0.756 | 0.746 | +0.005 |

Three findings. **At matched information the circuits match their
capacity-matched control** — within about 0.015 at every size, slightly
ahead at 2,000 rows. **The circuits gain almost nothing from fifty times
more data** (+0.007, +0.005), and neither does the linear control
(−0.001), while the tree ensembles convert the same data into +0.075 and
+0.084. **The ceiling is therefore capacity, not data**: the circuits
improve with data the way a linear model does, which locates the limit in
the expressive capacity of four qubits and two entangling layers rather
than in the corpus. The cost asymmetry is part of the result — 40 s and
251 s of simulation at the largest size, against 3 to 5 s for the tree
ensembles, to reach a lower score.

Two methodological notes belong with this table. Reaching the full corpus
required evaluating the circuit on whole batches by parameter broadcasting
rather than row by row, which is about 160 times faster; the earlier
row-by-row implementation is what had confined this stage to a few hundred
examples. And the test set must be held fixed: an interim version measured
small-sample models on a small test set and full-corpus models on a larger
one, and the circuits then appeared to *degrade* with more data. The
capacity-matched control fell by the same amount, which is what revealed
the error rather than the result.

Finally, the cautionary history is kept. On the earlier, artefact-
contaminated labels this stage appeared to show quantum matching a strong
classical model; the corrected task reversed that. The circuits had not
changed, the task had (L9).

### 5.5 The four variants, end to end

Mean over the seven plans; rank is of the ground truth among 512 to 4,000
candidates (median in brackets). Full tables:
`data/evaluation_accuracy.csv` and `evaluation_tests.csv`.

**Algorithmic weights** — the ground truth ranks deep in the field
(median ≈ 2,400 of ≈ 4,000) for every variant; top-1 and top-5 are zero
across the board; the judges, confined to a top 15 the ground truth never
reaches, can only change which *wrong* answer is picked (V2 improves
Chamfer slightly, 0.958 vs 1.000 m). The equal-weight score prefers
sparse regular grids; real derived arrangements are dense and irregular;
§5.3 said exactly this would happen.

**Learned weights** — a different regime:

| variant | mean rank (median) | top-1 | top-5 | Chamfer m | IoU |
|---|---|---|---|---|---|
| V1 (features) | 21.9 (10) | 0.000 | 0.429 | **0.248** | 0.704 |
| V2 (pictures) | 21.7 (9) | 0.143 | 0.429 | 0.264 | 0.718 |
| V3 (score only) | 21.7 (9) | 0.143 | 0.429 | 0.271 | **0.722** |
| V4 (quantum) | 22.0 (11) | 0.143 | 0.429 | 0.282 | 0.683 |

Under the learned weights the score alone puts the ground truth at
median rank 9, recovers it *exactly* (Chamfer 0, IoU 1.0) as its first
pick on one of seven plans, and lands within 0.28 m Chamfer on average.
On top of that, the judges move it by one or two places, in mixed
directions: the feature judge trades the top-1 away for the best Chamfer
of the four; the image judge trims the Chamfer distance slightly at the
cost of a little overlap; and the quantum judge, trained on the same 80 %
partition as the other two, gives the weakest geometric agreement of the
four.

**Paired tests**: no variant separates from V3 at n = 7 (every Wilcoxon
p = 1.000: each pair of variants differs on at most two of the seven
plans, in mixed directions). **That is the thesis's headline stated the
way the protocol demands: the score weights dominate; the judges are
second-order; no judge improves on the score alone in any consistent
direction.**

**One basis, not two.** An earlier revision also compared the variants on
a lateral drift index, which had the merit of not depending on the derived
labels. With that axis removed, the accuracy table above is the whole
comparison, and its dependence on the labelling procedure is measured only
indirectly, through the threshold sensitivity in Section 5.6.

### 5.6 Sensitivities

**Score weights (V3, algorithmic).** ±50 % on any single weight: Kendall
τ 0.85–0.92 against the nominal ranking, top-5 overlap 80–100 %.
Dropping any term entirely: τ 0.69–0.78 — every term carries real
ordering. One perturbation is diagnostic: dropping `t_density` improves
the ground truth's mean rank from 2,429 to 643 — the equal-weight
density term actively pushes the truth away, independently confirming
the learned vector's negative density coefficient.

**λ.** In algorithmic mode λ is inert (the ground truth never reaches
the shortlist the judge sees). In learned mode, λ from 0 to 1 moves the
mean rank from 21.7 to 21.9, trades one first-place recovery away
(0.143 → 0.000 at λ ≥ 0.3) and leaves the top-five rate unchanged
(0.429 throughout): real, small, and mixed in both directions. The
default 0.3 is defensible precisely because little hangs on it.

**The 160 mm threshold.** With ground truth re-derived from the demo
plans' own wall thicknesses (learned weights, V3): median-regime rank
≈ 24 at 140 mm and ≈ 22 at 160, then 90 at 180 and 195 at 200; Chamfer
0.232 → 0.271 → 0.431 → 0.612 m. The headline is stable across
140–160 mm and degrades beyond — partly because the corpus the weights
were learned from was itself derived at 160. Every number in this thesis
is conditioned on that constant; this table is the price tag.

**Score-judge overlap (L7).** Spearman ρ between the score's ranking and
ML1's over the full candidate set: mean 0.161, range 0.08–0.28. The
judge is nearly orthogonal to the score — it carries genuinely
independent information; it simply does not carry much that helps rank
the ground truth once the learned weights have done their work.

### 5.7 Speed

Parsing and enumeration are milliseconds; scoring ~4,000 candidates
takes ≈ 1–3 s; a judge re-ranks a shortlist in milliseconds (V2) to
~0.2 s (V4). Judge training: ~10 s (ML1), minutes (ML2), ~3–15 minutes
(quantum, simulator). The full 4 × 2 evaluation over seven plans with
every sweep runs in under half an hour on a laptop.

---

## 6. Discussion

**What the comparison actually found.** The four-way design was built to
ask "which judge?", and answered a prior question instead: *the weights
matter far more than the judges*. Learning five readable weights from
the corpus moved the ground truth from median rank ~2,400 to ~9;
swapping judges moved it by one or two places. No judge repays its cost
end to end; the image judge's advantage as a classifier — measured by the
wall-channel ablation — is that pictures carry the plan context that
scale-invariant features cannot. The registration signal (columns live
in load-bearing walls) shows up as the dominant learned weight and as
ML2's 0.995: two appearances of one fact, each carrying the same caveat
that the ground truth was derived from that very relation.

**The negative results are load-bearing.** V1 and V4 do not beat V3;
nothing reaches significance at n = 7; the equal-weight score is
actively wrong about density for wall-bearing buildings. Each of these
is reported as a result, not smoothed over — the protocol was designed
so that this outcome would be publishable rather than embarrassing.

**The evaluation now rests on one basis.** Removing the drift axis left
agreement with the derived labels as the sole criterion. That agreement is
what the thesis set out to measure, and the four-way comparison is intact,
but the pipeline no longer contains any check that is independent of its
own labelling procedure. The threshold sensitivity in Section 5.6 is the
nearest substitute, since it shows how far the results move when the
labelling assumption is varied, and reinstating a label-independent check
is the second item of further work.

**What did not work, kept on the record.** The earlier cost function's
sizing shortcuts (bending-only beam sizing, ignored buckling) remain
documented in the git history but left the ranking path entirely — a
cost model whose unit rates cannot be defended is not made defensible by
better sizing underneath it. The earlier quantum "match" evaporated with
the lattice artefact. The earlier CNN conclusion ("features win")
belonged to a different task and reversed on this one. And this
revision's own first evaluation run had a blend that muted the judges
entirely (normalising the penalty within the shortlist instead of the
full set); it was caught because all four variants returned
byte-identical tables, and the fix is documented in stage 12.

---

## 7. Limitations

The loophole register the refactor was driven by, with the closing state
of each. Items marked **open** are carried, not hidden.

1. **L1, no evaluation protocol** — closed: stage 12 is the spine.
2. **L2, the ground truth is derived, not observed** — open by nature,
   managed: the derivation is documented, externally sanity-checked
   (thickness distribution vs DISCS), and its constant swept — headline
   stable 140–160 mm, degrading at 180–200 (§5.6). Every accuracy in
   this thesis means "agreement with the derivation", nothing more.
3. **L3, circular negatives** — closed; the 0.947 → 0.862 drop is the
   receipt.
4. **L4, unreachable ground truth** — closed by aligning stage 04/05
   with the derivation; 7/7 and reported first.
5. **L5, min–max fragility** — closed (median/IQR everywhere).
6. **L6, learned-score tautology** — avoided by construction (pairwise
   ranking) and documented.
7. **L7, double counting** — measured: score-judge rank overlap ρ ≈ 0.15;
   the old second ranker retired; λ ablated. One residual overlap is
   stated: the ranker's negatives and the judges' negatives both contain
   lattice-like arrangements.
8. **L8, ML2 discards scale** — a stated trade, and now a measured one
   (the ablation shows what the picture buys instead).
9. **L9, the quantum comparison** — closed as feasibility with matched
   and full controls; single-split, stated.
10. **L10, position-blind drift** — no longer applicable: the drift axis
    has been removed from the thesis entirely. What replaces it is the
    absence of any structural check, recorded as limitation 17 below.
11. **L11, row-level leakage** — closed: grouped by plan in every model;
    scores re-reported after the change.
12. **L12, single seed** — closed for ML1 (×5), weights (×5), ML2 (×3).
    **Open** for the quantum circuits: one split per training size, so the
    small movements across sizes cannot be separated from run-to-run
    variation, which is why the sweep is read for its trend rather than
    its individual figures.
13. **L13, synthetic balance** — PR-AUC reported everywhere; prevalence
    at deployment unknowable and said so.
14. **L14, no cross-source evidence** — closed: LOSO both directions for
    both judges; ML1's asymmetric gap reported.
15. **L15, documentation drift** — closed by this revision: every stage
    README, EXPLAINED, notebook, the app and this document describe the
    same pipeline, and the root README carries the deletion changelog.
16. **L16, no external validity** — **open, and the primary further
    work**: no practising engineer has rated these arrangements. Until
    the 10–20-plan blind rating study is run, every metric here is
    self-referential — the models agree with a labelling procedure, not
    yet with a human.

Additional limits this work created: the evaluation set is seven plans
(paired tests run, power stated); inclined-wall candidates are disabled
in evaluation; CubiCasa's ground truth rests on a per-plan quantile
rather than a physical threshold; and enumeration past the cap is sampled,
rather than exhaustive.

17. **No measurement is independent of the labels.** With the drift axis
removed, every figure in Section 5 expresses agreement with the
arrangement that Section 3.1's procedure derives. If that procedure is
systematically wrong, no result here would reveal it. The threshold
sensitivity bounds the effect of one assumption within the procedure; it
cannot test the procedure itself. Reinstating a label-independent check,
whether structural or human, is the clearest way to close this.

---

## 8. Conclusions and further work

### Conclusions

1. **The candidate space can be made to contain the truth** — but only
   by aligning generation with the labelling procedure; reachability is
   a design property, not a given, and must be reported before accuracy.
2. **Five readable weights, learned by ranking, beat both an opaque
   second model and hand-set equality** — median rank of the truth: ~9
   versus ~2,400. For wall-bearing residential stock the designerly
   priors on density and regularity point the wrong way.
3. **Judges are second-order once the weights are right**: no judge
   improves on the score alone in any consistent direction. The
   appearance judge's advantage as a classifier is the plan context,
   proven by ablation, not the convolution.
4. **A 4-qubit circuit learns the task to matched-classical level and no
   further, and more data does not move it.** Across fifty times more
   training data it gains +0.007 while equally-informed tree ensembles gain
   +0.075, so its ceiling is expressive capacity rather than data. That is
   a sharper feasibility statement than a single small-sample score can
   support, and it still supports no claim of advantage.
5. **The strongest plausibility signal on this corpus is column-wall
   registration**, and every method allowed to see it finds it; a thesis
   built on derived labels must say, as this one does, that the signal
   is partly its own derivation reflected back.

### Further work, in order of value

1. **The expert-rating study (L16)**: 10–20 plans, each variant's top-1
   beside the derived truth, rated blind by a practising structural
   engineer.
2. **Recover CubiCasa's absolute scale** (standard door widths) so the
   160 mm rule applies physically to a second source.
3. **Grow the held-out set** beyond seven plans until the paired tests
   have power; the protocol already scales.
4. **Reinstate a label-independent check.** Any measurement that does not
   depend on the derived labels would restore what the removal of the
   drift axis took away; a lateral or gravity check with real member
   sizes would serve, as would the expert study in item 1.
5. **Learn the placement policy itself** (which nodes of a chosen grid
   carry columns) rather than choosing between two fixed policies.

---

## 9. How to reproduce all of it

```
pip install -r requirements.txt          # + shapely / pennylane / tensorflow as fenced

# corpus (needs the two archives from Zenodo; validates itself against
# the committed tables before overwriting anything)
python 02_data_pipeline/regenerate_corpus.py \
    --swiss data/swiss-dwellings-v3.0.0.zip --cubicasa data/cubicasa5k.zip

# models and results tables
python 07_layout_classifier/train_ml1.py
python 08_ml_ranker/train_learned_weights.py
python 10_quantum_ml/train_quantum.py
python 11_cnn_baseline/train_ml2.py
python 11_cnn_baseline/train_ml2_ablation.py

# the evaluation, all four variants, both score modes, every sweep
python 12_evaluation/evaluate.py --variants V1,V2,V3,V4

# the app
cd 09_full_pipeline && python app.py --variants V1,V2,V3,V4
```

Every notebook reads committed tables and runs without the raw archives.
Every figure in `figures/` is regenerated by the notebook of its stage.
The committed corpus, coordinates, walls, weights and every results table
are in `data/`, with small plain-text extracts in `data/samples/` for
reading in a browser.
