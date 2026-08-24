# Stage 11: The image judge

Estimates the probability that a candidate arrangement is plausible from a
picture of it, using a small convolutional neural network. It is trained on
the same examples and the same labels as the feature judge in Stage 07.
This is the judge used by pipeline variant V2, and the comparison between
the two judges is one of the thesis's principal experiments.

## Converting an arrangement to a picture

Each arrangement is drawn on a 64 by 64 grid of pixels in two layers: the
first holds the load-bearing walls as single-pixel lines, the second holds
the columns as three-by-three blocks. Everything drawn is scaled to fit the
picture, which removes absolute size entirely.

That scaling is the design argument for an image judge on this corpus.
Because the three sources are drawn in three unit systems, the feature judge
is restricted to six measurements that do not depend on scale; a picture
scaled to fit sidesteps the problem, since a plan in metres and a plan in
drawing pixels become the same kind of image. The same property is also its
cost: once everything is scaled to fit, the judge cannot distinguish a
six-metre span from a six-foot one. This is a deliberate trade rather than
an oversight.

Implausible examples are created by moving the column positions and *then*
drawing them. Adding noise to the pixels instead would reproduce, in image
form, exactly the circularity that Stage 07 removed from the feature judge.

## Results

Splits are grouped by plan over three random seeds. The comparison with the
feature judge is **paired**: that model is fitted to the measurements of
exactly the same rows and scored on exactly the same split. This corrects
two faults in the earlier version of this stage, in which the two models saw
different random examples and a resolution comparison altered three things
at once. Results are in `data/ml2_results.csv`.

| Model | ROC AUC | PR AUC |
|---|---|---|
| Image judge, walls and columns | **0.995 ± 0.003** | 0.995 ± 0.003 |
| Image judge, columns only (walls layer blanked) | 0.874 ± 0.019 | 0.819 ± 0.032 |
| Feature judge, paired on the same rows and splits | 0.860 ± 0.007 | 0.830 ± 0.010 |

Trained on one source and tested on the other:

| Model | Swiss to CubiCasa | CubiCasa to Swiss |
|---|---|---|
| Image judge, walls and columns | 0.983 | 0.988 |
| Feature judge, paired | 0.673 | 0.805 |

## What the blanked-walls test establishes

Removing the walls layer is the experiment this stage exists to perform.
Given the same information, the two judges are indistinguishable: 0.874 for
the picture against 0.860 for the measurements, with overlapping ranges.
Given the walls, the picture is far better, and the advantage survives
testing on a source the model never saw.

The conclusion is therefore about information, not architecture. The
convolutional network does not read column patterns better than a
gradient-boosted model reads measurements of them. The picture wins because
it can show whether the columns coincide with the walls, and the
scale-independent measurements cannot carry that relationship across
sources drawn in different units.

The accompanying caveat is the same one that appears wherever this signal
does: the derived labels place columns inside load-bearing walls by
definition, so the strength of the signal is partly a restatement of the
labelling procedure. The same qualification applies to the off-wall measure
and to its large learned weight in Stage 08. The thesis treats these as one
finding with three appearances: on this corpus, the strongest available
indication that an arrangement is real is whether its columns coincide with
its walls.

## A superseded conclusion

The earlier version of this stage reached the opposite result, reporting
0.841 for measurements against 0.725 for pictures. That experiment was a
different one: it showed the network grid lines only, with no walls layer
and with columns placed at every crossing, under the earlier labels. On grid
lines alone the measurements did win; on arrangements shown in their
architectural context the pictures do. Both results are retained in the
record with their dates.

## Files

- `cnn_baseline.ipynb` — the pictures the network receives, the three-way
  result, and the blanked-walls test
- `train_ml2.py`, `train_ml2_ablation.py` — the experiments
- `cnn_judge.keras` — the trained judge used by variant V2, regenerated
  automatically when absent
- tensorflow is this stage's dependency and is not imported by the pipeline
