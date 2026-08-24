# Stage 11, said simply

## What this stage does

Trains a second judge — same exam as stage 07's, different sense. The
feature judge reads six measurements *about* a layout. This one looks at a
64-by-64 picture *of* the layout: walls drawn in one colour layer, columns
as little squares in another. Same real examples, same fakes, so any
difference in score is a difference between the two ways of seeing.

## Why bother with pictures

The corpus mixes three drawing unit systems — metres, screen pixels,
roughly decimetres. Numbers like "average span" mean nothing across that
mix, which is why the feature judge is stuck with six carefully chosen
ratios. A picture dodges the whole problem: scale every plan to fit the
same small canvas and a metre plan and a pixel plan become the same kind
of image. The price is paid in the same coin: once everything is scaled
to fit, the picture cannot tell a 6-metre span from a 6-foot one. Chosen
trade, stated up front.

## What happened, in three numbers

* Pictures with walls visible: **0.995** — almost perfect.
* Pictures with the walls erased: **0.874**.
* Features (same layouts, same fakes, same split): **0.860**.

The middle number is the important one. Erase the walls and the picture
judge falls back level with the feature judge — so the convolutional
network is *not* better at reading column patterns than plain boosting is
at reading measurements of them. The entire advantage was the wall layer:
the picture judge can check whether the columns *sit on the walls*, and
every kind of fake — shaken, tilted, thinned, borrowed from another
building — breaks that registration.

Even trained on Swiss buildings only and tested on Finnish ones, the
walls-visible judge barely drops (0.98+). Walls-and-columns registration
apparently looks the same in every country's drawings.

## The grain of salt, served with the meal

Our "real" layouts were *derived* from load-bearing walls in the first
place — so "real columns sit on walls" is partly our own recipe read back
to us. The same note appears wherever this signal shows up (the off-wall
score term, its big learned weight). The honest summary of all three:
on this corpus, the one great tell of a real arrangement is that its
columns live in its walls, and every method that is allowed to see the
walls finds it.

## About the old conclusion

An earlier version of this stage found the opposite — features 0.84,
pictures 0.73 — and said so loudly. That experiment showed the network
grid *lines* only (no walls, and columns faked at every crossing), on the
old ground truth. Different pictures, different task, different answer.
The write-up keeps both results with their dates, because a thesis that
quietly deletes its old conclusions is worth less than one that explains
them.
