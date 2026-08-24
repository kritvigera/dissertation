# Stage 02 explained in plain words

## What this stage does

Takes three big collections of building plans, each stored in a completely
different format, and turns them all into the same simple thing: a list of
points where columns would go, plus about twenty numbers describing the shape.

Think of it as three people describing the same house in three languages. This
stage is the translator.

## Why it has to exist

The three collections were made by different people for different reasons.

* **ResPlan** stores plans as saved Python objects.
* **Swiss Dwellings** stores them as a giant spreadsheet where each wall is
  written as a line of text like `POLYGON ((1.2 3.4, 5.6 3.4, ...))`.
* **CubiCasa5K** stores them as SVG drawings, the same file type a web designer
  uses for logos.

You cannot learn anything from three formats at once. So everything is
converted into one shape first.

## Words you will meet

* **Wall segment.** A straight piece of wall, written as four numbers: the x
  and y of where it starts, and the x and y of where it ends.
* **Grid line.** An imaginary straight line running across the whole building.
  Real buildings have them, and columns sit on them.
* **Bay.** The gap between two neighbouring grid lines.
* **Feature.** A single number describing something about the layout, for
  example how uneven the bays are.
* **Deduplication.** Removing copies. If a building has ten identical floors,
  you only want to learn from it once.

## Step by step, what happens to one plan

### Step 1: straighten it

Some plans are drawn at an angle. A Swiss apartment might be rotated 23 degrees
because that is how the building sits on its street.

To fix that, the code looks at every wall, measures its angle, and works out
which direction most of the wall length points in. Then it rotates everything
so that direction is horizontal.

There is one trick here worth explaining, because it looks strange in the code.
Walls come in pairs at right angles. A wall at 90 degrees is really the same
direction as a wall at 0 degrees, just turned. So if you average angles
normally, a building with equal amounts of both gets an average of 45 degrees,
which is wrong. The fix is to multiply every angle by four before averaging and
divide by four afterwards. Multiplying by four makes 0 and 90 land in the same
place, so they stop cancelling each other out.

### Step 2: turn walls into grid lines

Every wall that runs along an axis votes.

* A wall running up and down votes for a vertical grid line at its x position.
* A wall running left and right votes for a horizontal grid line at its y
  position.
* A wall at a genuine angle, say 30 degrees, votes for nothing. It is not
  really either, and the earlier version of this code forced it into whichever
  bucket it leaned towards, which planted a grid line where no wall was.
  Anything more than 10 degrees off an axis is now left out of the vote and
  handled separately in stage 04.

Votes that land close together are merged into one line. Merging is weighted by
wall length, so a long structural wall pulls the line towards itself and a short
partition barely moves it.

Short walls do not vote at all. Anything under 5 percent of the building's size
is ignored, because a cupboard wall should not create a structural grid line.

### Step 3: put a column where a load-bearing wall actually is

This used to be "a column at every crossing": 8 vertical lines and 6 horizontal
lines gave 48 crossings and therefore 48 columns. Simple, and wrong. It meant
the layout the models learned from was invented rather than found, and it gave
plans 84 to 130 columns where a real structural grid has 20 to 40.

What happens now has two parts.

**First, thin walls are thrown away.** Anything under 160 millimetres is a
partition, not something holding the building up. That number is the minimum
for a 2-hour fire rating. Three separate sources put the floor at 140 mm: the
European concrete code, a survey of real Swiss buildings, and an inventory of
102 Swiss buildings measured from their drawings. 160 is a slightly cautious
cut above that.

**Second, a column only goes where a surviving wall really passes through.**
The grid is drawn from the thick walls alone, and then each crossing is tested:
does a load-bearing wall physically cover this point, allowing for where the
wall starts, where it ends and how thick it is? Only then does a column appear.
A short wall claims only the crossings it actually reaches.

Finally, any column the original drawing marked explicitly is added, whether or
not a wall covers it. Those are the only columns here that are genuinely
observed rather than worked out.

**Say plainly what this is.** Swiss homes are held up by their walls, not by a
frame of columns. So this does not uncover a hidden column layout, because
there isn't one. It converts a wall structure into the equivalent frame of
columns and beams, which is a standard engineering way of modelling such a
building. Useful, defensible, and not the same thing as the architect's
drawing.

### Step 4: describe the shape with numbers

About twenty numbers per plan. They fall into two kinds, and mixing them up
would be a serious mistake.

**Absolute numbers** are things like footprint width and average bay size.
These carry units. The problem: ResPlan measures in something like decimetres,
Swiss in metres, CubiCasa in screen pixels. A bay of "40" means three different
things in the three collections. So these numbers can only be compared inside
one collection, never across collections.

**Shape numbers** are ratios, so the units cancel out. How long the building is
compared to how wide. How uneven the bays are compared to their average. These
can be compared across all three collections safely, and they are the only ones
the learning stages are allowed to use.

### Step 5: throw away duplicates

Two plans can describe the same layout. A ten storey block has the same floor
ten times over.

The code makes a fingerprint of each layout: slide it to the origin so position
does not matter, round the numbers, sort them, and hash the result. Same
fingerprint means same layout, so keep only the first.

## What comes out

Two files, both committed so you never need the raw downloads.

| File | What it is | Size |
|---|---|---|
| `data/features_all.csv` | 29,279 rows, one per unique layout, about 20 numbers each | 4.5 MB |
| `data/real_grids.csv.gz` | 27,972 rows holding the actual grid line positions | 1.9 MB |

The second one stores each building's grid lines slid to the origin and
divided by the longer side, so a 12 m flat and a 40 m office become directly
comparable as shapes. Two newer files carry what these two cannot: the
ground-truth column coordinates themselves (`derived_columns.csv.gz`) and
each plan's load-bearing walls with thicknesses (`lb_walls.csv.gz`), which
is what the judges and the learned ranking weights train from. Thirty layouts are lost between the two files because
they have no interior grid line at all on one side, so there is nothing to make
alternatives from.

## ResPlan needs one extra library

ResPlan is stored as saved Python objects that only open if the geometry
library that made them is installed. That library is **shapely**.

The notebook now includes the loader that uses it. Shapely appears in exactly
one function, only to read that one archive, and nothing else in the project
imports it. If you want to rebuild ResPlan's share of the corpus, install
shapely, download the archive, and set `RESPLAN_ZIP` in the notebook.

The alternative was to leave a gap in the corpus and explain it, which is worse
than adding one clearly fenced dependency.

## Check it yourself

Open the notebook and run it. Without the raw archives it will say "no archives
configured", read the committed files, and show you the conversion working on
the small sample plan where you can see every line and count the columns by
eye. That last part is deliberate: the conversion is proved on geometry small
enough to check by hand before you trust it on 30,000 plans.
