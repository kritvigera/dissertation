# Stage 09: The complete pipeline

Connects the stages and exposes the four variants that the thesis compares.

```
              04 parse -> 05 enumerate -> 06 score (fixed or learned weights)
                                              |
        +-----------------+------------------+-----------------+
        |                 |                  |                 |
   V1: feature       V2: image          V3: no judge      V4: quantum
   judge (07)        judge (12)         (baseline)        judge (11)
        |                 |                  |                 |
        +-----------------+---------+--------+-----------------+
                                    |
                          chosen arrangement
```

All four variants share every earlier stage, so the comparison is
controlled: the same grid, the same candidates and the same score table.
Each judge receives the fifteen best-scoring candidates and re-orders them
by combining the score with its own opinion:

```
combined = normalised penalty + lambda x (1 - probability of plausibility)
```

The penalty is normalised across the whole candidate set, so that
differences within the shortlist are small and the judge has real influence.
Lambda defaults to 0.3 and is varied in Stage 12 rather than asserted.

**Variant V3 is the baseline that the other three must beat.** If V1, V2 and
V4 do not improve on it under the Stage 12 protocol, that is the result and
is reported as such. Every variant runs under both sets of score weights,
giving Stage 12 its four-by-two tables.

## Code

`pipeline.py` is the single implementation of the pipeline. The web
application, the stage notebooks and the Stage 12 evaluation all import it,
which prevents the copies from diverging. The `Pipeline` class trains or
loads the requested judges once at start-up; a subsequent run takes about a
second.

```python
from pipeline import Pipeline
pipe = Pipeline(variants=("V1", "V3"))        # V2 requires tensorflow,
result = pipe.run(walls, constraints,         # V4 requires pennylane
                  variant="V1", score_mode="learned", lam=0.3)
```

## Web application

```bash
python app.py                          # variants V1 and V3
python app.py --variants V1,V2,V3,V4   # all four
python app.py --fast                   # judges trained on a sample
```

The page at `http://localhost:8000` exposes the variant, the score weights,
lambda, the span limits and the storey count, and offers the seven held-out
dwellings alongside the synthetic office plans. For each answer it shows the
chosen arrangement, its five score measures normalised within the candidate
set, and the judge's probability where a judge was used. No prices appear
anywhere, because the pipeline computes none. See `WEBAPP.md`.

## Files

- `full_pipeline.ipynb` — the variants run end to end on a held-out
  dwelling and on the synthetic office plan
- `pipeline.py` — the implementation of Stages 04, 05 and 06, the three
  judges, and the variant selection
- `app.py`, `index.html` — the demonstration server and page
