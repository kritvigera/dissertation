# Stage 02: Data pipeline

Converts three public floor-plan datasets, each in a different format, into
one representation: a structural column arrangement and about twenty
geometric measurements per plan.

| Source | Plans kept | Native format | Units as drawn | Label confidence |
|---|---|---|---|---|
| ResPlan | 16,812 | pickled polygons | approximately decimetres | `none` |
| Swiss Dwellings | 8,262 | CSV of polygon geometry | metres | `verified` |
| CubiCasa5K | 4,205 | SVG drawings | drawing pixels | `relative-scale` |
| **Total** | **29,279** | | | 12,467 usable for training |

## Deriving the labels

An architectural drawing does not record a structural column layout, so
this stage derives one by a stated procedure:

1. **Discard every wall thinner than 160 mm.** This is the minimum
   thickness for a two-hour fire rating, so thinner walls are treated as
   room dividers rather than primary structure.
2. **Build the grid from the surviving walls only.** A structural grid
   should follow structural elements; using every wall places grid lines at
   room spacing rather than at structural spacing.
3. **Place a column at every grid crossing that lies inside one of those
   walls.** The test uses each wall's start point, end point and thickness,
   so a short wall claims only the crossings it actually reaches. Testing
   the grid line alone would allow a stub wall at one end of a building to
   claim crossings at the other.
4. **Add every column the drawing marks explicitly**, whether or not it
   falls inside a wall.

### What the result represents

Steps 1 to 3 produce an **equivalent-frame idealisation of a wall-bearing
structure**: a column layout that behaves like the wall system it replaces.
These buildings do not have a column grid to recover.

Badoux and Peter, surveying more than fifty Swiss buildings from original
construction drawings, report that in the dominant Swiss apartment type
"the partition walls are structural bearing walls" and that "frame-type
structural systems are an exception". The corpus agrees: its ratio of wall
area to floor area is about 24%, against roughly 15% for shear-wall
buildings and 8% for concrete-frame buildings in published surveys. Because
nearly every wall carries load, the 160 mm threshold selects the *primary*
walls rather than separating structure from non-structure.

Only step 4 is observed. Everything else is a modelling decision, recorded
here so that the two are not confused.

### The 160 mm threshold

| Source | Thickness |
|---|---|
| Eurocode 2, minimum for a reinforced load-bearing wall | 140 mm |
| Badoux and Peter, thinnest concrete wall found in the Swiss stock | 140 mm |
| DISCS building inventory, 10th percentile across 102 Swiss buildings | 140–150 mm |
| **Two-hour fire rating minimum, the value used** | **160 mm** |
| DISCS median structural wall | 200 mm |

The measured thicknesses validate against that inventory: a median of
190 mm against DISCS's 200 mm, and a 75th percentile of 291 mm against
their 300 mm. The additional mass below 150 mm in this corpus is the
partition population that DISCS excludes by design, since it records only
load-bearing and insulating layers.

Because every downstream result depends on this one number, Stage 12
re-derives the labels at 140, 160, 180 and 200 mm and reports how far the
headline accuracy moves.

## What each source supports

**Swiss Dwellings** is drawn in metres and stores walls as polygons, so
thickness is measurable and the 160 mm threshold applies as written. It
also marks columns explicitly, supplying step 4. This is the only source
where the procedure works exactly as specified.

**CubiCasa5K** also stores walls as polygons, but in drawing pixels with no
recoverable real-world scale, so an absolute threshold in millimetres is
meaningless. It instead keeps the thickest walls of each plan by
proportion. This is a genuine weakness and is why those plans carry a
different confidence label. Recovering scale from standard door widths
would resolve it and is left as further work.

**ResPlan** supports neither step 1 nor step 4: it declares one wall depth
for a whole plan, so there is no per-wall thickness to filter on, and it
marks no columns. Its plans remain on the earlier every-crossing procedure,
are labelled `none`, and are **excluded from every training set**. They
remain useful for descriptive statistics and for testing the parser against
a third drawing convention.

## Method

Every dataset loader yields `(plan id, geometry)` and one conversion routine
does the rest:

1. rotate the plan so that its dominant wall direction is horizontal, since
   Swiss plans are drawn at arbitrary angles;
2. reduce each wall polygon to a centre line and a thickness;
3. apply the 160 mm threshold, then group the surviving wall positions into
   grid lines, weighting each wall by its length and setting tolerances as
   a fraction of plan size. Only walls within 10 degrees of an axis
   contribute, because a genuinely angled wall is not straightened by any
   rotation and forcing it into the grouping would place a grid line where
   no wall exists;
4. place a column at every crossing contained by a load-bearing wall, then
   add the marked columns;
5. hash the arrangement after moving it to the origin, so that identical
   layouts drawn in different places collapse to one entry;
6. compute the geometric measurements.

## Outputs

`regenerate_corpus.py` reruns the whole derivation from local copies of the
raw archives and validates the result against the committed tables before
overwriting them. On the most recent run the regenerated Swiss plans matched
the previous corpus on all 8,260 shared identifiers, with grid-line counts
and regularity identical, and CubiCasa matched on all 4,205.

| File | Contents |
|---|---|
| `features_all.csv` | 29,279 rows, one per unique arrangement |
| `real_grids.csv.gz` | grid-line positions, scaled to shape |
| `derived_columns.csv.gz` | the ground-truth column coordinates |
| `lb_walls.csv.gz` | each plan's load-bearing walls and thicknesses |
| `threshold_sweep.csv` | corpus statistics at each candidate threshold |
| `demo_plans/*.json` | the seven held-out plans, with wall thicknesses |
| `demo_plans/ground_truth.json` | their labels, stored separately |

The coordinate files are new in this revision. Earlier runs kept only
summary measurements and grid lines, which meant the labels themselves were
unavailable and neither the judges nor the accuracy protocol could be built.

shapely is used for two purposes only, unpickling the ResPlan archive and
measuring wall thickness from polygons, and nothing downstream imports it.

## Angled walls

The rotation step handles a building drawn at an angle to the page. It does
not handle a building that genuinely contains angled walls, such as a
chamfered corner, because no single rotation straightens those. Such walls
are excluded from grid-line grouping rather than forced into it. Offering
column positions where a grid line crosses an angled wall is implemented in
Stage 04.

## Points to carry forward

- **Units.** Absolute measurements are comparable only within a source, and
  the 160 mm threshold applies only where the units are real.
- **Occupancy distinguishes the two procedures.** Under the earlier
  every-crossing procedure the column count equalled the number of grid
  crossings in every plan by construction. Under the derived labels the two
  differ almost everywhere: the median Swiss plan places columns at about
  70% of its crossings and the median CubiCasa plan at about 62%, while
  every ResPlan plan still shows exact equality. That equality is the marker
  of the earlier procedure and the reason those plans are excluded from
  training.
