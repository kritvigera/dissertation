# Data

Inputs, the committed corpus, and the results tables the stages produce.

Readers browsing on GitHub should start with [`samples/`](samples), which
holds small plain-text extracts of the compressed files.

## Inputs

| File | Contents |
|---|---|
| `demo_plans/` | Seven real Swiss plans held out of every training set. Each wall carries its measured thickness, and columns marked on the original drawing are listed separately, so the 160 mm load-bearing rule can be applied to these files directly. See [`demo_plans/README.md`](demo_plans/README.md) |
| `demo_plans/ground_truth.json` | The derived arrangement of each plan at 140, 160, 180 and 200 mm, stored apart from the plans so that the pipeline cannot read its own answers |
| `holdout_ids.csv` | The seven plan identifiers, removed by every training script |
| `train_test_split.csv` | The canonical 80/20 partition of the remaining plans, grouped by plan (seed 42). The judges the pipeline deploys train on the 80 % side only |
| `sample_plan.json` | A synthetic six-storey office, 36 by 24 m, with a central atrium kept clear of columns |
| `skew_plan.json` | The same office with a chamfered corner and a diagonal wall, to exercise the angled-wall path |
| `test_plan.dxf` | The same office as a DXF file, to exercise the DXF reader |
| `constraints.json` | User constraints: storeys, span limits, unit scale, merge tolerance, excluded regions |

## Corpus

| File | Rows | Used by |
|---|---|---|
| `features_all.csv` | 29,279 | Stages 03 and 07. Twenty-nine columns |
| `real_grids.csv.gz` | 27,972 | Stage 08 |
| `derived_columns.csv.gz` | 12,467 | Stages 07, 10 and 11. The ground-truth column coordinates of each plan |
| `lb_walls.csv.gz` | 12,467 | Stages 08 and 11. Each plan's load-bearing walls and thicknesses, in the same frame as the columns |
| `threshold_sweep.csv` | 7 | Stage 02. Corpus statistics at each candidate load-bearing threshold |
| `samples/` | extracts | Reading on GitHub |

All of these are produced by `02_data_pipeline/regenerate_corpus.py` from
the Swiss Dwellings and CubiCasa5K archives, which are large and are not
committed. The script validates its output against the committed tables
before overwriting them; on the most recent run Swiss matched on all 8,260
shared identifiers and CubiCasa on all 4,205. ResPlan rows are carried
across unchanged and are descriptive only, having no derived labels.

Because the three sources are drawn in different units, absolute
measurements are comparable only within a source. Anything that crosses
sources uses either scale-independent measurements (Stage 07) or pictures
scaled to a fixed canvas (Stage 12).

## Model and evaluation results

| File | Produced by | Contents |
|---|---|---|
| `learned_weights.json` | Stage 08 | The five learned score weights and their diagnostics |
| `rank_pairs_summary.csv` | Stage 08 | Ranking accuracy by split, source and alternative type |
| `ml1_results.csv` | Stage 07 | Feature judge: five grouped splits, cross-source transfer, single-measurement baselines, difficulty by example type |
| `ml2_results.csv` | Stage 12 | Image judge: three grouped splits, cross-source transfer, the paired feature control, the blanked-walls test |
| `quantum_comparison.csv` | Stage 11 | Quantum circuits against capacity-matched and full-capacity classical controls |
| `evaluation_accuracy.csv` | Stage 12 | Agreement with the labels, by plan, variant and weight set |
| `evaluation_tests.csv` | Stage 12 | Paired significance tests between variants |
| `evaluation_lambda.csv` | Stage 12 | The judge-influence sweep |
| `evaluation_weights_sensitivity.csv` | Stage 12 | Score-weight perturbation and removal |
| `evaluation_threshold.csv` | Stage 12 | Accuracy with labels derived at each threshold |
| `evaluation_overlap.csv` | Stage 12 | Correlation between the score's and the judge's orderings |
| `summary_by_source.csv` | Stage 03 | Per-source medians |

## Intermediate files

Safe to delete; regenerated on demand.

| File | Produced by |
|---|---|
| `parsed_grid.json` | The Stage 04 notebook, for the synthetic office plan |
| `arrangements.json` | The Stage 05 notebook, for the same plan |
| `.swiss_plans_cache.pkl` | `regenerate_corpus.py`, a parse cache; not committed |
