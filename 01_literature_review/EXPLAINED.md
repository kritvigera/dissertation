# Stage 01 explained in plain words

`README.md` in this folder is the formal chapter. This file says the same
things in ordinary language, with nothing assumed.

## What this stage is

A list of other people's work, and a note next to each one saying which of my
choices it backs up.

That is all a literature review is. People make it sound grander than it is. In
a thesis you have to show two things: that you know what already exists, and
that what you did is not a worse copy of it. A list of papers with a reason
attached to each does both.

## Words you will meet

* **Paper.** An article published in a journal or a conference. It has been
  read and checked by other researchers before publication.
* **Preprint.** The same thing, but put online before anyone checked it. arXiv
  is where most preprints live. Useful, but treat with a bit more care.
* **Dataset.** A pile of data someone collected and released for others to use.
* **Patent.** A legal claim on an invention. Not a paper, but it counts as
  prior work, and it is public.
* **DOI.** A permanent web address for a paper. If a paper has one, anybody can
  find it forever.
* **Prior art.** Anything that already existed before your work. If someone did
  your idea first, that is prior art, and you have to say so.

## What is in this folder

| File | What it holds |
|---|---|
| `references.csv` | 43 sources, one per row, in a table |
| `references.bib` | The same 43, in the format reference software reads |
| `literature_review.ipynb` | Code that checks the table and draws charts from it |
| `README.md` | The written chapter |

## The two columns that do the real work

Most reference lists have author, year, title and nothing else. This one has
two extra columns, and they are the point of the whole file.

**`justifies`** says which of my decisions the source supports. Not what the
paper is about. So the ISO 2848 entry does not say "a standard about modular
coordination". It says "this is why the modular_share feature works". When I
later ask "why did I do that?", the answer is already written down.

**`citation_status`** says whether I actually checked the entry against the
publisher's own record or a primary index. All 43 now say `verified`; the
handful that once said `partial` were either completed or removed.

That second column matters more than it looks. It is very easy to write a
reference from memory, get the year wrong, and have an examiner notice. Writing
down what you have not checked is safer than pretending you checked everything.

## The four groups of work I read

**Group one: floor plan collections.** Other people have collected thousands of
building plans and released them free. Three of them are used in this thesis.
Without these there is no project, because there is nothing to learn from.

**Group two: turning drawings into data.** Reading a picture of a plan and
working out where the walls are. This is a solved problem, so I did not redo
it. All three collections already give walls as lines with coordinates, so I
start after that step.

**Group three: using machine learning to design structure.** People have done
this. One group in China trained a model on real shear wall designs and it
worked, which proves the idea is sound. Most of the rest of the field predicts
how a finished design will behave rather than proposing a design, which is the
opposite end of the problem from this thesis.

**Group four: quantum machine learning.** Only relevant to folder 11.

## The uncomfortable discovery

The single closest piece of prior work is not a paper. It is a patent held by
Autodesk, filed in 2020 and granted in 2024.

It describes: draw grid lines so they line up with the load bearing walls, keep
the spacing between lines inside a minimum and a maximum, then put columns
where the lines cross, and mark some crossings as not allowed.

That is stages 04 and 05 of this thesis, almost word for word.

I found it after building the pipeline, not before. Two reasonable reactions.
The bad one is to leave it out and hope nobody looks. The good one, which is
what the chapter does, is to put it in the front and say plainly what is still
different:

* they use a learned agent to search, I check every option and so cannot miss
  the best one;
* their ranking rules were written by the authors, mine are learned from more than
  eleven thousand real buildings and tested in a way that cannot cheat;
* their data is not published, mine is;
* their final answer is checked against a set of rules; mine is measured
  against the arrangement the real building actually uses, on plans no
  model was trained on.

An examiner who finds this patent and sees it discussed on page one thinks you
are thorough. An examiner who finds it and sees no mention thinks something
worse. It is better to be the first kind.

## The mistake I corrected

An earlier version of this chapter claimed that nobody had ever learned
structural layout from real data. That was simply wrong. Liao and colleagues
did it in 2021 with real shear wall drawings.

So the claim had to shrink until it was true. The true version is: their
drawings were company property and were about shear walls in Chinese flats.
Nobody has tried using the free, public, architectural plan collections as
training data for structure. That is a smaller claim, and it survives contact
with the evidence, which is the only kind of claim worth making.

## Things I can now say with a source behind them

* Steel cost is mostly not the steel. Making and erecting it is about 70
  percent of the bill, and the connections between beams are 30 to 50 percent
  of the fabrication cost while being under 5 percent of the weight. This
  used to justify pricing layouts; after the prices were removed it does a
  more honest job — it says which *geometric* quantities the score should
  charge for: member count, off-grid columns, unrepeated bays. (AISC)
* Real buildings repeat a bay size because there is an international standard
  telling designers to work in multiples of 300 mm. So when my model looks for
  a repeated module, it is looking for something that genuinely exists.
  (ISO 2848)
* Splitting data at random when samples are related to each other makes your
  scores look better than they are. There is a whole literature on this in
  ecology, of all places. (Roberts and others)
* Learning by telling real things apart from made up things has a proper name,
  noise contrastive estimation, and it comes with a warning: you only learn the
  boundary against the fake data you made, so the fake data is a design choice.
  (Gutmann and Hyvarinen)
* On small tables of numbers, tree based models still beat neural networks. So
  not using deep learning here is a decision with evidence, not laziness.
  (Grinsztajn and others)

## What the notebook does

1. Loads the table and counts what is in it.
2. Lists the entries that are not fully checked, so they cannot be forgotten.
3. Sorts each source by how findable it is: a DOI, an arXiv number, a web
   address, a patent number, or nothing but a name. Only the two design
   standards fall in the last group, which is fine because they are sold rather
   than published free.
4. Draws two charts: how many sources per topic, and when they were published.
   Roughly half are from 2020 or later, which shows this problem only became
   workable recently.
5. Builds the decision register automatically from the `justifies` column, so
   the table in the chapter cannot drift out of date.
6. Writes `references.bib` for whatever you write the final document in, and
   marks the incomplete entries inside the file so the warning travels with
   them.

## What to do next with it

Close the five incomplete entries. Two need author lists, two need page
numbers, and the two design standards need their clause numbers checked in the
actual standards rather than in websites quoting them.

Then run a proper patent search. The closest prior work was a patent once
already, so it could easily happen again.
