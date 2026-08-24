# Predicting structural column arrangements from architectural floor plans

MSc thesis, Imperial College London.

When a building is first designed, someone must decide where its columns
will stand. The architectural drawing shows the walls and rooms; it does
not show the structure. This project automates a first proposal: given a
floor plan, it lists possible column arrangements, ranks them, and
measures how closely the highest-ranked arrangement matches the one the
real building uses.

The central experiment is a controlled comparison of four ways of doing
the ranking. One uses a fixed geometric formula alone. The other three add
a machine-learning model that re-orders the formula's best candidates: a
model reading numerical descriptions of each arrangement, a model reading
pictures of it, and a model running on a simulated quantum computer. The
formula-only version is retained as the baseline that the other three must
beat.

## Terms used throughout

| Term | Meaning |
|---|---|
| Column arrangement | The set of positions where columns stand on one floor |
| Load-bearing wall | A wall that carries the weight of the building above it, rather than merely dividing rooms |
| Grid line | A straight reference line running the width or depth of the plan; columns are placed where two grid lines cross |
| Span | The distance between two adjacent columns |
| Ground truth | The arrangement treated as the correct answer during measurement. Drawings do not record one, so it is derived from the load-bearing walls (Stage 02) |
| Judge | A machine-learning model that estimates how plausible an arrangement is |
| ROC AUC | A score from 0.5 (no better than guessing) to 1.0 (perfect) measuring how well a model separates two classes |

## Pipeline

```
architectural plan (.json / .dxf) + user constraints
  |
  [04] parser        walls -> grid lines -> candidate column positions
  [05] generator     every arrangement allowed by the span limits
  [06] score         five geometric measures -> one penalty per candidate
  |                  (weights either fixed or learned, Stage 08)
  +-- V1  [07] re-ranked by the feature judge
  +-- V2  [11] re-ranked by the image judge
  +-- V3       not re-ranked: the score's own order stands (baseline)
  +-- V4  [10] re-ranked by the quantum judge
  |
  best arrangement
  |
  [12] evaluation    the accuracy table, significance tests and
                     sensitivity analyses
```

The pipeline uses geometry only. It contains no member sizes, no material
prices and no load values, and it performs no structural analysis; Stage 06
explains why these were removed and the changelog below records where the
underlying reasoning now sits.

## Repository layout

| Folder | Contents | Notebook |
|---|---|---|
| `01_literature_review/` | 43 verified sources, the decision register, the novelty assessment and the closest prior work | `literature_review.ipynb` |
| `02_data_pipeline/` | Three public floor-plan datasets converted to one format; the derived ground truth and its sensitivity analysis | `data_pipeline.ipynb` |
| `03_exploratory_analysis/` | Descriptive statistics of the resulting corpus | `exploratory_analysis.ipynb` |
| `04_plan_parser/` | Walls to grid lines and candidate column positions | `plan_parser.ipynb` |
| `05_arrangement_generator/` | Enumeration of feasible arrangements | `arrangement_generator.ipynb` |
| `06_score_function/` | The five geometric measures and two sets of weights | `score_function.ipynb` |
| `07_layout_classifier/` | The feature judge, used by variant V1 | `layout_classifier.ipynb` |
| `08_ml_ranker/` | Learning the score weights from the corpus | `ml_ranker.ipynb` |
| `09_full_pipeline/` | The four variants wired together, and a web demonstration | `full_pipeline.ipynb`, `app.py` |
| `10_quantum_ml/` | The quantum judge, used by variant V4, with classical controls | `quantum_ml.ipynb` |
| `11_cnn_baseline/` | The image judge, used by variant V2 | `cnn_baseline.ipynb` |
| `12_evaluation/` | The evaluation protocol and its results | `evaluation.ipynb`, `evaluate.py` |
| `data/` | Inputs, the corpus, and all results tables | |
| `figures/` | Figures produced by the notebooks | |

`THESIS.md` presents the whole project as a single document.
`09_full_pipeline/WEBAPP.md` describes the interactive demonstration.

## Contribution

The claim was checked against literature published between 2020 and 2026;
Section 1.9 of the literature review records the evidence and Section 1.10
the closest prior work, including work that narrows the claim.

No individual component is new. Predicting column positions from plans has
been done with a convolutional network on synthetic data, and is claimed
in a granted patent. Learning structural layout from real buildings has
been done for shear walls. Quantum classification of buildings has been
done for post-earthquake safety assessment.

The contribution is the controlled comparison: a fixed geometric score, a
feature-based judge, an image-based judge and a quantum judge applied to
one task, one corpus and one set of labels, with the score-only pipeline
retained as a baseline, capacity-matched controls on the quantum
comparison, and paired significance tests on held-out plans. Two
supporting contributions are a documented and sensitivity-tested procedure
for deriving column-arrangement labels for public floor-plan datasets, and
a method for learning the score weights by ranking real arrangements
against alternatives drawn from the same plan.

## Principal results

- **Candidate coverage.** After the parser was aligned with the labelling
  procedure, the ground-truth arrangement appears among the generated
  candidates for all seven held-out plans. This figure caps every accuracy
  measurement and is therefore reported before any of them.
- **The weights matter more than the judges.** Learned weights rank the
  correct arrangement above an alternative in 89.4% of paired comparisons,
  against 74.7% for equal weights. Under learned weights the median rank
  of the ground truth among roughly 4,000 candidates improves from about
  2,400 to single figures.
- **The feature judge scores 0.862 ± 0.006 ROC AUC** once its training
  examples are generated by moving columns rather than by editing the
  numbers the model reads. The previous figure of 0.947 was inflated by
  that circularity, and the reduction is reported as a finding.
- **The image judge succeeds through context, not architecture.** It
  reaches 0.995 ROC AUC with the wall outline visible and 0.874 without
  it, against 0.860 for the feature judge. The advantage comes from
  information the numerical features cannot carry, not from the neural
  network itself.
- **The quantum judge is limited by capacity, not by data.** Trained on the
  entire corpus rather than a small sample, a four-qubit circuit gains only
  +0.007 ROC AUC from fifty times more data, exactly as the
  capacity-matched linear control does (−0.001), while tree ensembles given
  the same data gain +0.075. It matches that control at every training size
  and is beaten by models with real flexibility. No advantage is claimed.
- **No variant beats the baseline significantly** on seven plans (every
  paired p = 1.000). The full tables are in `data/evaluation_*.csv`.

## Changelog: what was removed

| Removed | Previously | Where the reasoning now sits |
|---|---|---|
| Beam and column sizing, and the steel section catalogues | Stage 06 sized every member in order to price it; Stage 09 re-checked those members | The span-demand measure in Stage 06: span squared multiplied by supported width, which is the standard bending formula with its constant removed. The structural logic is preserved; the catalogue is not needed |
| Gravity load values and material strengths | The inputs to that sizing | Removed from the ranking path by design. A future member-checking stage would reinstate them |
| All prices (steel per kilogram, per-column and per-section penalties, labour rates) | Stage 06 reported a cost in pounds | The density, off-wall and repetition measures in Stage 06, which capture the same cost drivers without assigning money to them |
| The member-by-member Eurocode check, and the lateral drift index that briefly replaced it | Stage 09, now removed | Nothing. The pipeline performs no structural analysis, and accuracy against the derived labels is the sole basis of evaluation. The consequence is stated in the limitations |
| The second ranking model | Stage 08 narrowed a shortlist using a further learned model | Retired to avoid counting one signal twice. Stage 08 now learns the score weights |
| The `fill_ratio` feature | A column in the corpus table | Removed, being computable from three columns already present. The evidence it carried is stated directly from those counts (Stage 03) |
| Training examples created by editing features | The feature judge's negative class | Replaced by examples created by moving column positions, shared by all three judges (Stage 07) |

## Installation

```bash
pip install -r requirements.txt
jupyter notebook
```

The pipeline requires numpy, pandas, matplotlib, seaborn, scikit-learn and
scipy. Three further packages are used by single stages and are not
imported by the pipeline itself: shapely (Stage 02, measuring wall
thickness in the raw datasets), pennylane (Stage 11, the quantum judge)
and tensorflow (Stage 12, the image judge).

## Running the code

Every notebook reads the committed data tables and runs without the raw
datasets. To reproduce the results tables:

```bash
# corpus, labels and sensitivity analysis (requires the two raw archives)
python 02_data_pipeline/regenerate_corpus.py \
    --swiss data/swiss-dwellings-v3.0.0.zip --cubicasa data/cubicasa5k.zip

python 07_layout_classifier/train_ml1.py          # feature judge
python 08_ml_ranker/train_learned_weights.py      # score weights
python 10_quantum_ml/train_quantum.py             # quantum judge and controls
python 11_cnn_baseline/train_ml2.py               # image judge
python 11_cnn_baseline/train_ml2_ablation.py      # its wall-outline ablation
python 12_evaluation/evaluate.py --variants V1,V2,V3,V4

cd 09_full_pipeline && python app.py --variants V1,V2,V3,V4   # web demo
```

## Data

| Dataset | Plans with derived labels | Units as drawn |
|---|---|---|
| Swiss Dwellings | 8,262 | metres |
| CubiCasa5K | 4,205 | drawing pixels of unknown scale |
| ResPlan | none; descriptive statistics only | approximately decimetres |

The raw archives are large and are not committed. The repository holds the
feature table, the grid lines, the ground-truth column coordinates and the
load-bearing walls of every labelled plan, which is sufficient to retrain
every model. The seven demonstration plans carry wall thicknesses, and
their labels are stored separately so that the pipeline cannot read its own
answers. See `data/README.md`.

## Principal limitations

Stage-specific limitations appear in each folder, and `THESIS.md`
Section 7 lists all of them. Four are central:

1. **The labels are derived, not observed.** They represent an idealisation
   of buildings whose walls carry the load, and they depend on a 160 mm
   wall-thickness threshold. The threshold is varied between 140 mm and
   200 mm and the effect on accuracy is reported, but no engineer has
   reviewed the derived arrangements. That review is the first item of
   further work.
2. **One signal dominates every learned component**, namely whether
   columns coincide with load-bearing walls. Because the labels were
   derived from that same relationship, its strength is partly circular.
   This is stated wherever the signal appears.
3. **Seven plans are few.** Paired significance tests are used, and their
   limited power at this sample size is reported rather than glossed over.
4. **Angled walls are excluded from the evaluation**, because the labelling
   procedure cannot represent a column that stands off the rectangular
   grid. The angled-wall path is exercised only in demonstrations.
