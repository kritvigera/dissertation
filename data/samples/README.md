# Previewable samples

The full data files are either large or compressed, so GitHub does not
render them in the browser. These are small plain-text extracts of the same
files, showing the shape of the data without any download.

| Sample | Extracted from | Contents |
|---|---|---|
| `corpus_summary.csv` | `features_all.csv` | One row per source: plan count and medians |
| `features_all_sample.csv` | `features_all.csv` | The first 25 rows, all 29 columns |
| `real_grids_sample.csv` | `real_grids.csv.gz` | The first 25 grids, as line positions |
| `derived_columns_sample.csv` | `derived_columns.csv.gz` | Ten plans' ground-truth column coordinates |
| `lb_walls_sample.csv` | `lb_walls.csv.gz` | Five plans' load-bearing walls with thicknesses |

These are extracts, not inputs. No stage of the pipeline reads them.

## Reading the full files

```python
import pandas as pd
features = pd.read_csv("data/features_all.csv")          # 29,279 rows
grids    = pd.read_csv("data/real_grids.csv.gz")         # 27,972 rows
columns  = pd.read_csv("data/derived_columns.csv.gz")    # 12,467 rows
walls    = pd.read_csv("data/lb_walls.csv.gz")           # 12,467 rows
```

`real_grids.csv.gz` stores each plan's grid lines as space-separated numbers
in two text columns; unpack one with
`np.fromstring(row.x_lines, sep=" ")`. The lines are shifted to the origin
and divided by the longer side, so only the shape remains, which is what
allows plans drawn in three different unit systems to be compared.

`derived_columns.csv.gz` stores each plan's ground-truth column coordinates
as `"x y;x y;..."` in that source's own units, and `lb_walls.csv.gz` stores
its load-bearing walls as `"x1 y1 x2 y2 thickness;..."` in the same frame of
reference. Both unpack with
`[[float(v) for v in p.split()] for p in text.split(";")]`. These two files
are what allow the judges to train on real coordinates and the score weights
to be learned by ranking real arrangements against alternatives.
