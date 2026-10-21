# Assignment 2 Report

*Delete this italic guidance as you fill in each section. You'll be asked to
defend any of this without your code in front of you. Write only what you
can actually explain.*

- **Name**:
- **Student ID**:
- **Email**:
- **Group**: [BBADBA 5A | BBADBA 5B | PPLEDBA 5A | BDBA 3A]

## Business problem, and the cost of being wrong

> DRAFT: to be reviewed and adjusted. Every figure here lives in `config.py`,
> so changing one there changes the pipeline and the dashboard with it.

**The decision.** A fruit packing line photographs every piece of fruit on the
conveyor. For each piece someone decides: ship it, or throw it out. Today that
decision is made by line graders looking at the fruit as it passes. The model
gives each piece a probability that it is rotten, and that probability sends it
to one of three outcomes:

- **pass**: the piece goes into a retail pack and is shipped;
- **review**: the piece is diverted to a grader at the grading table, who
  picks it up, turns it and decides;
- **fail**: the piece is knocked off into the waste/juice bin.

The line supervisor owns the two thresholds. The graders handle what lands in
review.

**What each mistake costs (per piece).**

| outcome | cost | why |
|---|---|---|
| Rotten piece shipped (false negative) | **€4.00** | The rot is found in a retail pack. The retailer refunds or writes off the pack (a 1 kg pack is about €2–3 at retail), mould spreads to the neighbouring pieces, and the retailer charges the packer back for handling the quality claim. €4 sits in the middle of that range. |
| Fresh piece discarded (false positive) | **€0.30** | The wholesale value of one piece of fruit we could have sold (about €1.20–1.80/kg wholesale at roughly 5–6 pieces/kg). Nothing else happens. |
| Rotten piece discarded (true positive) | €0 | It was worth nothing before the camera saw it. Throwing it out is the correct, free outcome. |
| Fresh piece shipped (true negative) | €0 | Business as usual. |
| One review | €0.015 | 3 seconds of a grader at a fully loaded €18/hour. Graders are assumed to catch 97% of rotten pieces and wrongly bin 2% of fresh ones. |

**The asymmetry.** Shipping one rotten piece costs about **13 times** as much as
throwing away a good one. So the model should throw out a piece well before it
is "more likely rotten than not". If the scores were calibrated to production,
the cost-neutral cut would be at P(rotten) ≈ 0.30 / (0.30 + 4.00) ≈ 0.07, not
0.5. This is why the threshold is chosen by sweeping cost, not by accuracy.
The metric we care about most is **recall on the rotten classes**: a fresh
apple confused with a fresh orange costs nothing, but a rotten piece called
fresh costs €4.

**Volume and base rate.** One line handles about 2 pieces per second over an
8-hour shift, which is 57,600 pieces a day. We assume **4%** of incoming pieces
are really rotten. The dataset is far richer in rotten fruit than that, so all
costs are re-weighted to the 4% base rate rather than read from the test split.
Two graders, each with 7 productive hours, can review at most 16,800 pieces a
day: 29% of the line's volume. The review band has to fit inside that.

## Dataset

> DRAFT: to be reviewed and adjusted.

- **Roboflow Universe**: workspace `rishwanths-workspace`, project
  `freshness-of-fruits`, **version 3**, image classification, CC BY 4.0.
  https://universe.roboflow.com/rishwanths-workspace/freshness-of-fruits/dataset/3
- Version 3 has no Roboflow preprocessing or augmentation, so the images are
  the uploaded originals at their own sizes (1,451 distinct sizes, around
  350–560 px).
- **One image = one piece (or bunch) of fruit**, labelled with one of six
  classes: `freshapples`, `freshbanana`, `freshoranges`, `rottenapples`,
  `rottenbanana`, `rottenoranges`. The three `rotten*` classes are the
  positive class: the pieces that must not ship.

**Why it serves the problem.** It labels exactly the call a grader makes,
fresh or rotten, for the three fruits a line would handle, and it keeps the
fruit type. That lets us check whether the model fails more on one fruit than
another, rather than only on average.

**How it differs from the problem.** These are web and stock photos, not
conveyor photos. Fresh fruit is mostly glossy studio shots on white, often
with leaves or water droplets. Rotten fruit is mostly obviously rotten
(mould, black bananas). Early bruises, the hard case on a real line, are rare.
See "What the dataset's composition implies".

**Other candidates rejected** (checked through the Roboflow API):

| Dataset | Why not |
|---|---|
| `anil-ozkan2/meyve` v1 | 3,347 originals inflated to 8,033 by mosaic/crop augmentation |
| `zahrabahadori/fresh-and-rotten-fruit` | Messy classes with only 31–77 images for some fruits |
| `11thsacred/fruit-freshness-detection` | 96 images |
| `project-ta/fruit-freshness-detection` | Object detection, not classification |
| `university-of-mines-and-technology/fruits` | Unnamed classes `0` / `1` |

## What the dataset's composition implies

> DRAFT: to be reviewed and adjusted. Numbers from `artifacts/dataset_stats.json`.

**The published splits leak.** Most of the 8,195 images were already
augmented by the dataset's original author: their filenames carry
`rotated_by_75_`, `vertical_flip_`, `saltandpepper_`, `translation_`. Stripping
those prefixes leaves only **1,513 distinct source photos**, about 5.4 images
per photo. In the published split, **99.1% of validation and 99.4% of test
images are a copy of a photo that is also in train.** A test score on that
split would mostly measure whether the model remembers a photo it has seen
rotated. So the pipeline re-splits the downloaded images **by source photo**:
- per class, with a fixed seed;
- every copy of a photo in the same split;
- 70/15/15 by image count.

The dataset is used as published (not forked or edited on Roboflow); only our
local train/val/test assignment changes. Leakage on the split we use is 0%.

| split | images | distinct photos | fresh apple | fresh banana | fresh orange | rotten apple | rotten banana | rotten orange | rotten share |
|---|---|---|---|---|---|---|---|---|---|
| train | 5,702 | ~1,049 | 851 | 808 | 763 | 1,266 | 1,160 | 854 | 57.5% |
| val | 1,251 | 230 | 185 | 178 | 171 | 273 | 253 | 191 | 57.3% |
| test | 1,242 | 234 | 185 | 174 | 165 | 275 | 254 | 189 | 57.8% |

(Published split, not used: train 6,165 / valid 1,223 / test 807.)

**What follows from it:**

- **The real sample size is the number of photos, not images.** The test set
  has 1,242 images but only 234 independent photos, about 39 per class. A
  per-class recall measured on 39 photos has a 95% interval of roughly
  ±10–15 points. Per-class differences smaller than that are noise.
- **Classes are reasonably balanced** (763–1,266 training images per class),
  so no class is too thin to train. Fresh oranges and fresh bananas are the
  smallest classes.
- **Rotten is 57% of the data but about 4% of a real line.** So accuracy on
  this test set says little about cost on the line. Every cost is re-weighted
  to the 4% base rate (see Cost matrix section). The threshold is then chosen
  on validation and only reported on test.
- **Object size**: this is classification. The fruit fills most of the frame
  in nearly every image, so object size is not a concern. Framing would be
  much the same on a conveyor camera.
- **The test split does not look like production.** Studio "fresh" photos
  versus amateur "rotten" photos invite a shortcut: the model may learn the
  photo style instead of the rot. And the rotten examples are mostly far past
  the borderline. So the test numbers below are an **upper bound** on what
  the model would do on a real line. They cannot tell us how it handles a
  small bruise under conveyor lighting, which is exactly the case that costs
  money.

## Finetuning and baselines

*Which model and task (detection or classification) and why. A results table on
the same test split: your finetuned model, the pretrained COCO model zero-shot,
and a trivial baseline. What changed, and whether the gap is as large as you
expected.*

## Per-class evaluation and failure modes

*Per-class AP50, or per-class precision/recall and the confusion matrix.
Which class carries the headline number, which one is dragging, and what the
model actually confuses with what. Show where it breaks, not just that it
works.*

## Cost matrix and operating point

*Your euro cost for each kind of error and where each figure came from. The
base rate you assumed. The cost-against-threshold sweep, and the operating
point you chose. State the assumption that would most change your answer if it
were wrong.*

## What a person still has to check

*Your two thresholds, and what share of cases they send to a person. How many
that is per day, how long one takes to check, and how many people that needs.
Is that realistic? Then your design priced against the three alternatives:
doing nothing, having a person check everything, and letting the model decide
on its own.*

## Limitations & next steps

*Real limitations you found, and concretely how you'd address each one with
more time or data. Not generic hedging.*

## Generative AI use disclosure

*Per the syllabus AI Policy: what you used and how, or "no AI content used."*
