# Stage 01: Literature review

Forty three sources, each verified against the publisher record or a primary
index. `references.csv` is the machine-readable version,
`literature_review.ipynb` audits it and maps every source onto the design
decision it supports, and this chapter sets out the argument.

The scope is deliberately narrow. This is not a survey of machine learning in
civil engineering; it is the evidence base for a specific set of choices made
in Stages 02 to 12. Every section closes by naming the decision it justifies,
or the decision it fails to justify.

---

## 1.1 Question and search strategy

**Research question.** Given an architectural floor plan, can a structural column
arrangement be proposed, ranked and evaluated automatically, learning what counts
as a plausible grid from real buildings rather than from hand written rules?

**Search.** Queries were run in August 2026 across Google Scholar, arXiv, ACM DL,
ScienceDirect, Springer, IEEE and Google Patents, seeded with the terms
*floor plan dataset*, *floorplan vectorisation*, *structural layout generation*,
*column placement*, *shear wall layout*, *structural surrogate model*,
*topology optimisation deep learning*, *noise contrastive estimation*,
*learning to rank*, *variational quantum classifier*. Backward snowballing was
used from the two reviews (`sun2021mlreview`, `malaga2022opinionated`) and from
the closest prior art (`wang2024columnrl`, `liao2021shearwallgan`).

**Inclusion.** A source is included only if it does one of three things:
supplies data the thesis actually consumes, establishes a method the thesis
actually uses, or contests a decision the thesis actually makes. Papers that are merely adjacent
were left out, which is why the list is forty three items and not four hundred.

**Verification.** Every entry has been checked against the publisher or
repository record for authors, year, venue and identifier, and every entry in
`references.csv` now reads `verified`. One source that could not be verified,
a 2024 paper whose author list could not be traced, was removed rather than
carried with a caveat. The design standards are cited by number and year only,
since the documents themselves are paywalled and their clause numbering should
be confirmed against a purchased copy before being quoted.

---

## 1.2 Floor plans as a source of structural data

Large vector floor plan corpora now exist and are open. Three are used here.
CubiCasa5K (`kalervo2019cubicasa`) contributes 5,000 densely annotated plans as
SVG polygons. Swiss Dwellings (`standfest2022swiss`) contributes roughly 42,500
apartment models with walls stored as geo-referenced geometry, released CC-BY-4.0
by Archilyse and quality assured to a median area deviation of 1.2 percent.
ResPlan (`abouagour2025resplan`) contributes 17,000 residential plans in vector
form with metric coordinates.

Two further corpora were considered and rejected. RPLAN (`wu2019rplan`) is the
most cited residential corpus but is raster first and oriented towards room
semantics, so metric wall geometry has to be recovered before anything structural
can be said. MSD (`vanengelenburg2024msd`) is derived from Swiss Dwellings and
would have duplicated a source already in the corpus.

Recovering geometry from drawings is itself a mature field. Liu et al.
(`liu2017raster2vector`) framed raster to vector conversion as junction
prediction followed by integer programming, and Zeng et al.
(`zeng2019deepfloorplan`) improved wall and room recognition with a multi task
network. This thesis deliberately does not compete there: all three sources are
consumed as vectors, so the parsing problem starts one step later, at walls to
*structural* grid rather than pixels to walls.

**What this justifies.** Stage 02's choice of three sources and the decision to
work from vector geometry. **What it does not justify** is treating those corpora
as structural ground truth, which section 1.6 returns to.

---

## 1.3 Learned structural layout design, and the closest prior art

Three strands matter, and they are not equally close to this work.

**Response prediction.** The largest strand predicts structural behaviour from a
given design, usually by training a surrogate on simulated results. Sun, Burton
and Huang (`sun2021mlreview`) survey it and show that prediction and assessment
dominate, while proposing a layout is thinly covered. This thesis deliberately
does not sit in that strand: performance is computed here, not predicted.
Malaga-Chuquitaype (`malaga2022opinionated`) is more pointed, arguing that the
field has been readier to automate the parts engineers do not mind losing than
the judgement they exercise, and warning against models asked to certify what
they cannot explain. That warning shapes the division of labour here: the learned
stages shortlist, and analysis decides.

**Generative structural design.** Liao et al. (`liao2021shearwallgan`) trained
GANs on real shear wall design documents, abstracted and parameterised by height
and seismic category, and produced usable layouts. Lu et al. (`lu2022physicsgan`)
then added physics to the generator rather than trusting it alone. This strand is
the strongest evidence that structural layout *is* learnable from real design
data. Two limits matter for this thesis. The training documents are proprietary
design archives, not open data, so the work is not reproducible outside the group
that holds them; and the task is shear wall layout in Chinese residential
practice, not column grid inference from an arbitrary architectural plan.

**Column placement specifically.** The closest prior art is not a paper but a
patent. Wang and Nourbakhsh at Autodesk (`wang2024columnrl`, filed October 2020,
granted March 2024) claim a two agent reinforcement learning system in which the
first agent lays gridlines to maximise overlap with load bearing walls subject to
a minimum and maximum beam span, and the second places columns at gridline
intersections under rules that mark locations as required, preferred, candidate
or forbidden.

That is the same decomposition this thesis uses in stages 04 and 05: walls
vote for grid lines, span limits bound the search, columns stand at
crossings, and some crossings are excluded. The patent was found after the
pipeline had been built, and is reported here rather than omitted. It is
strong external evidence that the decomposition is the natural one, and it
sharpens what remains to contribute:

| | `wang2024columnrl` | This thesis |
|---|---|---|
| Search | Two RL agents, learned policy | Exhaustive enumeration over interior line subsets (seeded sampling past a stated cap), every candidate buildable by construction |
| Ranking | Reward terms defined by the authors | Four rankers compared under one protocol: a dimensionless algorithmic score (weights fixed or learned by pairwise ranking from 11,098 real plans) with three learned judges ablated against the score-only baseline |
| Data | Not disclosed | Two open corpora with derived ground truth, regenerable end to end, coordinates committed |
| Reproducibility | Patent, no released code or data | Open notebooks, committed feature and coordinate tables |

**What this justifies.** The stage 04 and 05 decomposition, and the decision to
put the contribution in the ranking and the evaluation rather than in the search.
Their "required" locations also justify stage 04's treatment of columns the
drawing itself marks: they are input the enumerator must honour, not part of
the answer.

---

## 1.4 Topology optimisation, and why it is the wrong frame here

Deep learning has been used extensively to accelerate topology optimisation.
Sosnovik and Oseledets (`sosnovik2019nn4topopt`) treated the iteration as image
to image translation; Yu et al. (`yu2019neartopo`) removed the iteration
altogether and predicted a near optimal topology directly; and a recent review
(`topopt2023review`) maps the field, including the retraining cost that makes
such models brittle when the design domain changes.

This is the closest neighbouring literature, and it is the wrong tool for
this problem. Topology optimisation returns a density field over a continuum. A
column grid is a discrete, dimensionally coordinated object that has to be built,
detailed and connected, and the operation that turns a density field into a
buildable grid is precisely the hard part, which the density formulation does not
address. Enumerating discrete arrangements under span constraints keeps every
candidate buildable by construction.

**What this justifies.** Stage 05's exhaustive discrete generator over a
generative continuum model. **What it costs** is scalability, and section 1.9
returns to it.

---

## 1.5 Screening with a cheap model and confirming with an expensive one

The funnel shape, many candidates screened cheaply and a few confirmed
expensively, is not invented here. It is the standard structure of surrogate
assisted optimisation, surveyed by Jin (`jin2011surrogate`): use a cheap model
to filter, and spend the real evaluation budget on the survivors.

What this thesis takes from that literature is the architecture, not the
practice of predicting performance — and one deliberate departure from it. In
surrogate-assisted optimisation the learned model stands *inside* the search,
replacing the expensive evaluation. Here the algorithmic score and the learned
judge are two *independent* rankers of the same candidate set, combined by an
ablated blend and — crucially — compared against the score-only baseline
(variant V3). No published head-to-head of that kind was found for building
structural layout (section 1.9), which is why the evaluation, not the search,
carries the contribution.

Danhaive and Mueller (`danhaive2021subspace`) supply the complementary
argument. Optimisation alone, they show, systematically misses the qualitative
properties designers care about, which is why performance conditioned
exploration beats pure search. That is the case for the judges existing at all
beside the score function.

**What this justifies.** The score-then-judge arrangement, with the judge
confined to the top fifteen candidates, and the decision to place the
contribution in the ranking and its evaluation rather than in the search.

## 1.6 How the learned components are set up

Four choices in the judges and in the learned score weights follow directly
from the literature.

**Learning from constructed negative examples.** No dataset labels a layout
as implausible, so the judges construct their negatives: real arrangements
are the positive examples and altered copies of those same arrangements are
the negatives. Training a model to separate them teaches it what real
arrangements look like. The method is noise contrastive estimation
(`gutmann2010nce`), and naming it matters because it carries a warning: what
the model learns is the boundary between the real data and *the particular
negatives that were constructed*. How the alteration is performed is
therefore a design decision rather than an implementation detail. This is
the reasoning behind the change described in Stage 07: negatives are now
produced by moving column positions, not by editing the measurements the
model reads.

**Ranking within a group.** Stage 08 does not ask whether an arrangement is
real in isolation. It asks which of several arrangements on one plan is the
one that plan uses. That is learning to rank (`liu2009ltr`), in which a set
of candidates belongs to a single query; here the plan is the query. It is
also the only formulation available: fitting a model to the fixed score's
output would be circular, and reusing the judges' examples would make the
score and the judge the same model.

**Splitting the data so the reported score is honest.** If rows are split at
random, a plan's real arrangement can fall in the training set while an
altered copy of that same arrangement falls in the test set. The model
recognises the copy and scores well without having learned anything general.
Ecologists encountered this problem with samples taken close together, and
Roberts and colleagues (`roberts2017cv`) showed that random splits inflate
scores and that data should be split by whatever the samples have in common.
Kaufman and colleagues (`kaufman2012leakage`) give the general name,
leakage. Every model here is therefore split by plan, and the split is
repeated over several seeds. Both choices reduce the reported figures and
make what remains worth quoting.

**Choice of model.** Gradient boosting (`friedman2001gbm`), in its fast
histogram form (`ke2017lightgbm`). On tables of a few thousand rows, tree
ensembles still outperform neural networks (`grinsztajn2022tabular`), so
declining to use deep learning for the tabular judge is an evidence-based
decision. The image judge is a small convolutional network because its input
is a picture, not a table. The implementation is scikit-learn
(`pedregosa2011sklearn`).

**The limit that none of these sources closes.** These plans are
architectural. What the parser recovers is a building's wall pattern, not an
engineer's structural grid, so what the learned components measure is
agreement with the way real buildings divide a floor. That is a reasonable
proxy and it is not the same thing as structural merit. No source in this
review closes that gap, and the thesis states it as a limitation.

---

## 1.7 Why the ranking is geometric, and what the cost literature still justifies

An earlier revision of stage 06 was a cost function in pounds: sections from a
catalogue, steel priced per kilogram, £350 per off-grid column, erection billed
by the crew-hour. It was deleted, and the reason belongs in the review: a cost
model is only as defensible as its unit rates, and those rates were indicative
figures with no auditable source. The industry evidence that motivated the
cost function was never the problem — what it actually establishes is *which
geometric quantities drive the bill* (`aisc_cost`):

* making and erecting the steel is about 70 percent of a steel package, and
  connections drive 30 to 50 percent of fabrication cost at under 5 percent of
  the weight — so the member count, which scales with the column count, is a
  first-order driver. That is the score function's `t_density`, unpriced;
* off-grid columns cost disproportionate detailing effort — `t_off_wall`;
* bending demand grows with the square of the span — `t_span_demand`, which
  keeps the w·L²/8 physics of the deleted beam sizing up to a constant;
* ISO 2848 (`iso2848`) makes modular coordination a published standard, so
  repeated bay sizes are a real habit of real buildings and repeated formwork
  a real economy — `t_repetition`.

The same evidence therefore applies at a higher level of abstraction: the
score retains the cost drivers and discards the currency. UK guidance (`sci_grids`) still supplies the
span sanity band for offices (7.5 to 9 m economical spacing), which sets the
app's default span limits; the evaluation on dwellings uses a wider band for a
reason stage 12 documents.

---

## 1.8 Quantum machine learning, and the limits of what can be claimed

Folder 11 runs the stage 08 task on a quantum circuit. This literature is what
keeps the claims in bounds.

**The model type is established.** Havlicek and colleagues (`havlicek2019`)
introduced both the variational classifier and the quantum kernel and ran them
on real hardware. Schuld and colleagues (`schuld2020circuit`) defined the
circuit design this folder starts from. Schuld and Killoran
(`schuld2019featurehilbert`) explain why encoding a number as a rotation angle
is a feature map, which is why the inputs are scaled into a bounded angle
range. PennyLane (`bergholm2018pennylane`) is the software.

**And one paper supplied the second architecture.** Perez-Salinas and
colleagues (`perezsalinas2020reupload`) showed that feeding the data into the
circuit again at every layer, rather than once at the start, makes a small
circuit far more expressive. Folder 11 runs both the plain and re-uploading
architectures; on an earlier, artefact-contaminated ground truth the
re-uploading model appeared to match strong classical baselines, and on the
corrected task it does not. The change is reported as such, because it is
itself evidence of how readily comparisons of this kind can mislead.

**The nearest applications are not design.** Quantum machine learning has
reached structural engineering as post-earthquake safety classification —
Bhatta and Dang (`bhatta2024quantum`) train a variational quantum classifier
on tabular building-and-demand features, the closest single work to folder
11's — and as structural health monitoring, while quantum treatment of
structural *geometry* exists only as annealing-based truss optimisation and
VQE-style analysis. The field's own review (`ploennigs2026quantumcivil`)
catalogues no learned plausibility judgment of a design-stage layout, which
bounds what folder 11 can and cannot claim to be first at (section 1.9).

**The limits are just as established, and they set the size of the circuit.**
McClean and colleagues (`mcclean2018barren`) showed that for many circuit
designs the training signal shrinks exponentially as you add qubits, until the
model cannot tell which way to adjust itself. That is why this stays small.
Huang and colleagues (`huang2021powerdata`) showed that classical models with
data are competitive even on problems built to favour quantum ones.

**And the benchmark sets the standard of fairness.** Bowles, Ahmed and Schuld
(`bowles2024benchmark`) tested 12 quantum models on 160 datasets and found
classical models generally ahead, and that removing entanglement often helped.
Their paper is as much about how to compare fairly as about the result, and
folder 11 follows it: same rows, same features, same split, classical baselines
run on the identical subsample.

---

## 1.9 What is new here

The claims below were checked against the 2020–2026 literature in August 2026
(searches over arXiv, Crossref, Semantic Scholar, publisher sites and Google
Patents; the closest works, including those that weaken the claims, are in
section 1.10). Each claim is stated as narrowly as the evidence supports.

**1. The controlled four-way comparison. This is the central claim.**
*Existed:* every pair of judge families, somewhere. Rule-based scores versus
feature ML for seismic screening; a non-learned metric versus deep models for
floor-plan similarity (`vanengelenburg2023ssig`); a variational quantum
classifier versus classical tabular models for post-earthquake building
safety (`bhatta2024quantum`); quantum versus classical CNNs for damage
imagery.
*New:* no study was found that applies even three of the four families, namely
an algorithmic score, feature-based learning, image-based learning and quantum
learning, to **one identical design task with one corpus and one set of
labels**. None retains the algorithm-only path as a baseline that every
learned variant must beat, provides capacity-matched controls for the quantum
comparison, or applies paired significance tests across held-out plans. That
controlled comparison, rather than any individual component, is the central
contribution.

**2. Column-arrangement ground truth derived from open architectural
collections, with its sensitivity measured.**
*Existed:* prediction of column layouts from plans, but trained either on
synthetic frames (`ampanavos2021approxiframer`) or on private archives
(`pizarro2021walls`, for shear walls). Open floor-plan corpora carry no
column annotations beyond the sparse column entities in Swiss Dwellings.
*New:* a documented derivation, applying a 160 mm load-bearing threshold,
placing columns at contained grid crossings and adding those the drawing
marks, applied to two open corpora. The coordinates are committed, the seven
evaluation plans are held out, and the one modelling constant is varied
between 140 and 200 mm so that its effect on every downstream figure can be
read directly.

**3. Score weights learned by ranking real arrangements against enumerated
alternatives.**
*Existed:* evaluation of generated structural layouts by expert-assigned
weights (`wang2025experteval`), by discriminators operating inside
generative training (`lu2022physicsgan`), and plausibility classifiers using
real examples as positives outside the construction domain
(`liu2024diffpop`).
*New:* a paired ranking model in the structural domain, taking the real
derived design as the positive example and alternatives generated on the
same plan's grid as negatives. The model is linear in the differences
between measures, so the learned weights can be read directly against the
fixed ones, and the two are compared under the same protocol as everything
else.

**4. Feature judge versus image judge on identical rows and labels.**
*Existed:* comparisons of representation between *generative* models, such
as graph against raster, and comparisons of tabular models against
convolutional networks on floor plans for the prediction of rent rather than
plausibility.
*New:* the same positive examples and the same negatives, with one model
reading six scale-independent measurements and another reading 64 by 64
pictures. A test that blanks the wall layer distinguishes two explanations
of the result: that convolution reads column patterns better, or that the
picture carries architectural context the measurements cannot.

**5. A quantum judge of design-stage arrangement plausibility.**
*Existed:* variational quantum classification of buildings, for
post-earthquake safety from tabular measurements (`bhatta2024quantum`);
quantum treatment of structural geometry, as truss optimisation by annealing
and as eigenvalue analysis; and a review of the field
(`ploennigs2026quantumcivil`) that records no application to design-stage
layout.
*New:* so far as a systematic search establishes, the first variational
quantum classifier applied to the plausibility of structural arrangements
derived from real floor plans. It is presented strictly as a feasibility
study, with capacity-matched classical controls, because four simulated
quantum bits and a few hundred examples cannot support a claim of advantage
in either direction (`bowles2024benchmark`, `huang2021powerdata`).

**6. An evaluation protocol that reports its own ceiling.**
*Existed:* splits taken at the level of individual rows, negatives produced
by altering the measurements a model reads, and accuracy tables that do not
state whether the reference answer was available to be found.
*New:* negatives produced by moving column positions, shared by all three
judges; splits grouped by plan and repeated over several seeds, with
precision-recall figures reported beside ROC AUC; transfer tested by
training on one source and testing on the other; and the coverage rate,
being whether the derived arrangement is present among the candidates at
all, reported before any accuracy figure. Candidate generation was aligned
with the labelling procedure until that rate reached seven of seven.

**What is not claimed as new.** The decomposition from walls to grid lines
to crossings is claimed in the Autodesk patent (`wang2024columnrl`), and is
reported in Section 1.3. Screening candidates cheaply and confirming a few
expensively is standard practice (`jin2011surrogate`). Predicting column
layouts from plans with a convolutional network has been done
(`ampanavos2021approxiframer`). Learning structural layout from real
buildings has been done (`liao2021shearwallgan`, `pizarro2021walls`). Every
learning method used here is a standard one.

---

## 1.10 Closest prior work

The five works nearest to this thesis, and what each does not do.

**Wang and Nourbakhsh, US Patent 11,941,327 B2 (`wang2024columnrl`).** The
same decomposition, as a patent: grid lines placed on load-bearing walls,
columns at crossings, positions marked as required, preferred or forbidden,
and a reinforcement-learning agent making the selection. No data is
disclosed, no comparison between learned and rule-based selection is made,
and no evaluation protocol is given. This thesis contributes the open
corpus, the derived labels and the four-way comparison, none of which has an
equivalent in the patent. It removes any claim to the decomposition itself.

**Ampanavos, Nourbakhsh and Cheng, 2021 (`ampanavos2021approxiframer`).** A
convolutional network that generates column positions from plan sketches,
and the closest published work to this task. It was trained on roughly
137,000 synthetic rectangular frames and evaluated by positional error
against synthetic labels. This thesis differs on the two points that matter
here: its labels derive from real buildings, and it ranks candidates rather
than generating them. The paper removes any claim that predicting columns
from plans with a convolutional network is new.

**Pizarro, Massone, Rojas and Ruiz, 2021 (`pizarro2021walls`).** Prediction
of shear-wall layouts from architectural plans using convolutional and
generative models, trained on 165 real Chilean projects, and the nearest
precedent for using real buildings as supervision. The data is private, the
subject is wall panels rather than columns, and the approach generates a
layout without enumerating or ranking candidates. The paper removes any
claim that learning structural layout from real buildings is new.

**Wang, Yu, Chen, Liao, Li, Hu, Tan and Lu, 2025 (`wang2025experteval`).**
The current state of the art in *selecting among* candidate structural
layouts: an expert-weighted evaluation of automatically generated shear-wall
schemes, combined with automated finite-element analysis. Its evaluator
applies expert-assigned weights rather than a model learned from data, and no
learned judge is compared against that evaluator on the same candidates. That
comparison is precisely what this thesis performs.

**Bhatta and Dang, 2024 (`bhatta2024quantum`).** A variational quantum
classifier distinguishing safe from unsafe buildings using tabular
measurements, compared against ten classical models on identical data. This
is the closest quantum work, and the reason claim 5 above is worded as
design-stage arrangement plausibility rather than quantum learning applied to
buildings in general. Their task is the assessment of an existing structure
after an event; it involves no algorithmic score, no image-based model and no
design candidates.

---

## 1.11 The gap this thesis sits in

The gap must be stated precisely, because the general form of the claim is
false. It is not true that structural layout has never been learned from real
data: Liao and colleagues (`liao2021shearwallgan`) and Pizarro and colleagues
(`pizarro2021walls`) have done so. The claim that survives the evidence is
narrower, and has four parts.

1. **Open architectural collections have not been used as structural training
   data.** They are large, free and reproducible, and they are used for room
   layout research. Learned structural design uses private archives, or
   geometry generated for the purpose.
2. **The problem of obtaining column-layout labels has not been solved
   publicly.** Architectural drawings do not record a column layout, so
   anyone learning columns from plans must either generate the answer
   synthetically (`ampanavos2021approxiframer`), hold a private engineering
   archive (`pizarro2021walls`), or derive a label. No derivation, with its
   assumptions and their sensitivity, has been published for the open
   corpora. Stage 02 is this thesis's answer, and its 160 mm threshold is
   varied rather than asserted.
3. **Candidate rankers are not compared with one another.** Generated or
   enumerated structural layouts are evaluated by rules, by finite-element
   analysis, or by expert weighting (`wang2025experteval`), but no study was
   found that ranks one fixed candidate set with both an algorithmic score
   and independent learned judges and reports whether the judges improve on
   the score alone. The four-variant protocol is this thesis's answer.
4. **Exhaustive checking does not scale, and no remedy has been
   established.** Exhaustive search is exact on simple plans and infeasible
   on heavily subdivided ones. Sampling beyond a stated limit, as in
   Stage 05, is a mitigation rather than a solution, and the generative
   literature scales but does not produce buildable grids.

Items 1, 2 and 3 are where this thesis contributes. Item 4 remains open and
is presented as such.

---

## 1.12 Decision register

Every non-obvious decision in the pipeline, with the evidence supporting it.
Entries marked *none* are those the thesis must defend on its own reasoning.

| Stage | Decision | Evidence |
|---|---|---|
| 02 | Three open vector corpora, ResPlan, Swiss, CubiCasa | `abouagour2025resplan`, `standfest2022swiss`, `kalervo2019cubicasa` |
| 02 | RPLAN rejected as raster first | `wu2019rplan` |
| 02 | Start from vector walls, do not re-solve recognition | `liu2017raster2vector`, `zeng2019deepfloorplan` |
| 02 | The 160 mm load-bearing cut, swept 140/160/180/200 | fire-rating minimum; *sensitivity measured by this thesis* |
| 04 | Walls vote for gridlines, columns at intersections | `wang2024columnrl` |
| 04 | Over-segmentation is the primary limitation | *none, this thesis measures it* |
| 05 | Exhaustive enumeration under span limits | `wang2024columnrl`, contrast `sosnovik2019nn4topopt`, `yu2019neartopo` |
| 05 | Discrete arrangements over a density field | `topopt2023review` |
| 05 | Office span defaults of 6 to 12 m around the economical band | `sci_grids` |
| 05 | Seeded sampling past the cap, never truncation | *none, argued in stage 05* |
| 06 | The score keeps the cost drivers and drops the unit rates | `aisc_cost` |
| 06 | Repetition as a term: modular coordination is a standard | `iso2848` |
| 06 | Span demand as the load-free bending index (w·L²/8 minus the constant) | *none, argued from first principles in stage 06* |
| 06 | Median/IQR normalisation, not min–max | *none, argued (loophole L5)* |
| 06+07 | Cheap score, judged shortlist | `jin2011surrogate` |
| 07 | Learn plausibility from real plans, not rules | `liao2021shearwallgan`, `danhaive2021subspace` |
| 07 | Negatives generated in geometry space, never in feature space | `gutmann2010nce`, `kaufman2012leakage` |
| 07 | Scale invariant features only, given mixed units | *none, forced by the corpora* |
| 07 | Gradient boosting for tabular features | `ke2017lightgbm`, `grinsztajn2022tabular`, `friedman2001gbm` |
| 07+08+12 | Grouped split by plan, repeated over seeds | `roberts2017cv`, `kaufman2012leakage` |
| 08 | Weights learned by pairwise ranking, real versus enumerated alternatives | `liu2009ltr` |
| 08 | Never fitted to the algorithmic score's output | *none, argued (loophole L6)* |
| 11 | Angle embedding plus strongly entangling layers | `havlicek2019`, `schuld2020circuit`, `schuld2019featurehilbert` |
| 11 | 4 qubits, 2 layers, no scaling up | `mcclean2018barren` |
| 11 | Identical data and features in both arms | `bowles2024benchmark` |
| 11 | No advantage claim from the result | `huang2021powerdata`, `bowles2024benchmark` |
| 12 | The image judge and the feature judge see identical rows and labels | `bowles2024benchmark` |
| 12 | Wall-channel ablation to locate the CNN's advantage | *none, this thesis measures it* |
| 13 | Reachability reported before any accuracy figure | *none, argued (loophole L4)* |
| 13 | Paired tests across plans, never side-by-side point estimates | *standard practice* |

---

## 1.13 Limits of this review

1. **The search covered English-language and mainly open-access sources.**
   Generative structural design is strongly represented in Chinese practice
   and journals, and the coverage here rests on the internationally published
   subset of that work.
2. **Patents were searched only through Google Patents, and only in English.**
   Since the closest prior art proved to be a patent rather than a paper,
   patent coverage matters more here than it usually would.
3. **This is a narrative review, not a systematic one.** It was assembled
   around an existing design rather than following a protocol such as PRISMA,
   and is therefore exposed to the bias of having searched for support.
   Section 1.3 is a partial corrective: the closest prior art is reported even
   though it anticipates part of the approach.
4. **The four-way novelty claim rests on absence of evidence.** Searches
   across several databases found no comparable study, but no search
   establishes that none exists. The claim is worded to be falsifiable by a
   single counter-example, and Section 1.10 names the work that would come
   closest to providing one.

## How to extend `references.csv`

One row per source. Fill `justifies` with the decision it supports, not a summary
of the paper, since that column is what the decision register is built from. Keep
`theme` and `method` vocabularies small or the coverage plots stop being
readable. Set `citation_status` to `verified` only after checking the publisher
record.
