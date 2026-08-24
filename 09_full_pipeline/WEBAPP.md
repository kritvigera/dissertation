# The web app

A page you can open in a browser, change the inputs, and watch the pipeline
rerun. Useful for a demo, a viva, or just for getting a feel for how the
answer moves when the rules change.

## Running it

```bash
cd 09_full_pipeline
python app.py
```

Then open <http://localhost:8000>.

Three flags:

| Flag | What it does |
|---|---|
| `--variants V1,V2,V3,V4` | which variants to prepare; the default `V1,V3` needs only scikit-learn, V2 needs tensorflow, V4 needs pennylane |
| `--fast` | trains the judges on a sample of the corpus, ready quickly |
| `--port 8010` | use a different port if 8000 is busy |

Startup trains (or loads) the requested judges once; after that every run
of the pipeline takes about a second, so dragging a slider updates the
answer almost immediately.

## What you can change

| Control | Effect |
|---|---|
| Floor plan | the synthetic office, the same plan from DXF, the office with inclined walls, or any of the seven held-out Swiss dwellings |
| Variant | V1 feature judge, V2 image judge, V3 score only, V4 quantum judge |
| Score weights | algorithmic (equal) or learned (stage 08's pairwise-ranking vector) |
| Storeys | 1 to 20 |
| Minimum / maximum span | the user's span band, in metres; picking a dwelling auto-widens it, because load-bearing walls sit far closer than office columns |
| Wall merge tolerance | how close two walls must be before they share a grid line (dwellings use the corpus "auto" rule) |
| Weight on the judge (λ) | how much the judge's distrust counts against the score; 0 turns every variant into V3 |
| Atrium checkbox | turns the office's no-column zone on and off |

## What you see

**The funnel** across the top: candidate positions, buildable layouts,
shortlist, answer. Narrow the span range and the layout count collapses;
set it too narrow and nothing survives, and the app says so instead of
failing.

**The drawing.** Grey lines are the architect's walls and **orange lines
are inclined walls**. Small dots are candidate positions — dark ones sit
inside a wall, pale ones do not. A red circle is a position blocked by
the atrium. Black squares are the chosen columns, and orange diamonds are
chosen columns sitting on an inclined wall.

**The cards.** Variant and weights, columns per storey, the score penalty,
the judge's plausibility (V3 shows "no judge"), the placement policy, and
the run time.

**The shortlist table.** The fifteen survivors in the variant's final
order, with the score penalty and the judge's opinion. Switching between
V3 and V1 shuffles the order, which is the judge disagreeing with the
score and the reason the comparison exists.

**The term chart.** The winner's five score terms, normalised within the
candidate set: which of density, span demand, irregularity, off-wall and
repetition actually won it the top spot.

## Things worth trying in a demo

1. **Switch V3 → V1 → V2 on a dwelling.** Same candidates, same score;
   watch what each judge promotes into first place.
2. **Flip the weights from algorithmic to learned** on a dwelling. The
   learned vector likes denser, wall-hugging layouts — stage 08 explains
   why.
3. **Set λ to 0, then 1.5.** At 0 every variant collapses into V3; at 1.5
   the judge overrules the score. The default 0.3 sits where stage 12's
   ablation put it.
4. **Drag the maximum span down towards the minimum.** The candidate count
   falls to nothing and the app explains why rather than crashing.
5. **Turn the atrium off** on the office plan. A column reappears in the
   middle and the funnel grows.
6. **Switch to the inclined-wall plan.** Crossings appear as open
   circles; any the winner uses become orange diamonds.

## How it is built

* `pipeline.py` holds the stages as ordinary Python functions and is the
  one canonical copy — the notebooks and the evaluation import it too.
* `app.py` prepares the judges once, then serves the page and answers
  requests at `/api/run`.
* `index.html` is one file with the styling and the drawing code inside
  it. No frameworks, no libraries loaded from the internet, no build step.

The server is Python's own `http.server` from the standard library, so
the project rule of numpy, pandas, matplotlib, seaborn and scikit-learn
only is not broken by adding a demo. V2 and V4 are opt-in precisely
because tensorflow and pennylane are fenced exceptions.

## The API, if you want to script it

```bash
curl -X POST http://localhost:8000/api/run \
  -H "Content-Type: application/json" \
  -d '{"plan":"demo_01","variant":"V1","score_mode":"learned",
       "min_span":0.5,"max_span":8,"merge_tol":"auto",
       "lambda_blend":0.3}'
```

You can also post your own plan instead of a built-in one, by sending a
`walls` list where each wall is `[x1, y1, x2, y2]` or
`[x1, y1, x2, y2, thickness]`.

The reply holds the parsed grid, the funnel counts, the shortlist with its
penalties and judge opinions, the winning layout with its columns and
score measures, and the time each stage took.
