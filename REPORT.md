# Assignment 2 Report

- **Name**: Sebastián Delgado-Aparicio
- **Student ID**:
- **Email**: sdelgadoapar.ieu2022@student.ie.edu
- **Group**: [BBADBA 5A | BBADBA 5B | PPLEDBA 5A | BDBA 3A]

## Business problem, and the cost of being wrong

I picked a decision that people make by eye thousands of times a day: on a
fruit packing line, is this piece of fruit good enough to ship, or does it go
in the bin? Today a line grader makes that call while the fruit rolls past. I
want a camera over the conveyor, and a model that gives every piece a
probability of being rotten. That probability sends the piece to one of three
places:

- **pass**: it goes into a retail pack and ships;
- **review**: it is pushed into a tray at the grading table, where a grader
  picks it up, turns it over and decides;
- **fail**: it is knocked off into the waste or juice bin.

The line supervisor sets the two thresholds, and the graders handle whatever
lands in review.

The two mistakes are very different in price, which is what makes this
problem interesting:

| outcome | cost per piece | where the number comes from |
|---|---|---|
| Rotten piece shipped (false negative) | **€4.00** | The rot is found in a supermarket pack. The retailer refunds or writes off the whole pack (a 1 kg pack sells for about €2–3), the mould has often spread to the pieces next to it, and the retailer charges the packer for handling the quality claim. €4 sits in the middle of that range. |
| Fresh piece binned (false positive) | **€0.30** | We threw away a piece we could have sold. Wholesale fruit is about €1.20–1.80 per kg, at five or six pieces per kg. Nothing else happens. |
| Rotten piece binned (true positive) | €0 | It was worth nothing before the camera saw it. Binning it is the right, free outcome. |
| Fresh piece shipped (true negative) | €0 | Business as usual. |
| One human review | €0.015 | 3 seconds of a grader's time at a fully loaded €18 an hour. I assume graders catch 97% of rotten pieces and wrongly bin 2% of fresh ones; they are good, not perfect. |

So shipping one rotten piece costs about **13 times** as much as binning a good
one. That ratio drives everything that follows:

- **It sets where the cut should be.** A piece should be binned once its chance
  of being rotten passes 0.30 / (0.30 + 4.00), about 7%, not the default 50%.
- **It sets what to measure.** Recall on the rotten classes matters more than
  overall accuracy. Mixing up a fresh apple and a fresh orange costs nothing;
  calling a rotten orange fresh costs €4.
- **It sets how to choose the threshold.** I sweep cost, rather than picking the
  threshold that looks best on accuracy.

For the volume, I assume one line at 2 pieces per second over an 8-hour shift:
**57,600 pieces a day**. I assume **4%** of incoming fruit is actually rotten.
Two graders work the shift with about 7 productive hours each, so the line can
review at most 16,800 pieces a day, 29% of its volume. Whatever I send to
review has to fit inside that.

## Dataset

**Roboflow Universe:** workspace `rishwanths-workspace`, project
`freshness-of-fruits`, **version 3**. It is an image classification project,
CC BY 4.0, used exactly as published (not forked):
https://universe.roboflow.com/rishwanths-workspace/freshness-of-fruits/dataset/3

Each image shows one piece (occasionally a bunch) of fruit, labelled with one
of six classes: `freshapples`, `freshbanana`, `freshoranges`, `rottenapples`,
`rottenbanana`, `rottenoranges`. Version 3 has no Roboflow resizing or
augmentation, so the images are the originals at their own sizes: 1,451
different sizes, mostly 350–560 px wide.

**Why this dataset.** It labels exactly the call a grader makes, fresh or
rotten, for three fruits a real line would handle. Because it keeps the fruit
type, I can check whether the model is worse on one fruit than another instead
of only looking at an average.

**Alternatives I compared** (through the Roboflow API) and rejected:

| dataset | why I did not use it |
|---|---|
| `anil-ozkan2/meyve` v1 | 3,347 real images inflated to 8,033 with mosaic and crop augmentation |
| `zahrabahadori/fresh-and-rotten-fruit` | Messy class names, and only 31–77 images for some fruits |
| `11thsacred/fruit-freshness-detection` | Only 96 images |
| `project-ta/fruit-freshness-detection` | Object detection, not classification |
| `university-of-mines-and-technology/fruits` | Classes named just `0` and `1` |

**Where it does not match the problem.** These are web and stock photos, not
conveyor photos. Fresh fruit is mostly glossy studio shots on white, often with
a leaf or water droplets. Rotten fruit is mostly very obviously rotten: thick
mould, black bananas. The hard case on a real line, an early bruise, barely
appears. I come back to this in every section below, because it limits what
the numbers can tell us.

## What the dataset's composition implies

**The first thing I found was that the published splits leak.** Most of the
8,195 images were already augmented by the dataset's original author, and the
filenames give it away: `rotated_by_75_…`, `vertical_flip_…`, `saltandpepper_…`,
`translation_…`. Strip those prefixes and only **1,513 distinct photos** remain,
about 5.4 copies of each.

In the split as published, **99.1% of validation images and 99.4% of test
images are a copy of a photo that is also in training.** A test score on that
split would mostly measure whether the model recognises a photo it has already
seen, just rotated.

So my pipeline re-splits the downloaded images by source photo:
- every copy of one photo lands in the same split;
- the split is done per class, with a fixed seed;
- the shares are 70 / 15 / 15 by image count.

Nothing on Roboflow is changed; only my local assignment of images to splits.
After this, no validation or test photo shares a source with training. As a
second check, I hashed every image and compared pixels: only 5 of the 1,242
test images (0.4%) still look near-identical to a training image.

| split | images | distinct photos | fresh apple | fresh banana | fresh orange | rotten apple | rotten banana | rotten orange | rotten share |
|---|---|---|---|---|---|---|---|---|---|
| train | 5,702 | 1,049 | 851 | 808 | 763 | 1,266 | 1,160 | 854 | 57.5% |
| val | 1,251 | 230 | 185 | 178 | 171 | 273 | 253 | 191 | 57.3% |
| test | 1,242 | 234 | 185 | 174 | 165 | 275 | 254 | 189 | 57.8% |

The published split (not used) was train 6,165 / valid 1,223 / test 807.

What I take from this:

- **The real sample size is photos, not images.** The test set has 1,242
  images but only 234 independent photos, around 39 per class. A recall
  measured on 39 photos carries an uncertainty of roughly ±10 points, so I
  will not read anything into per-class differences smaller than that.
- **No class is too thin to train on.** The classes are reasonably balanced,
  from 763 to 1,266 training images. Fresh oranges and fresh bananas are the
  smallest.
- **Rotten is 57% of the data but about 4% of a real line.** Accuracy on this
  test set therefore says little about cost on the line. I compute every cost
  from the model's rates on rotten and on fresh fruit separately, then
  re-weight them to 4%.
- **Object size is not a concern.** This is classification and the fruit fills
  most of the frame in almost every image, which is also how a camera mounted
  over a conveyor would frame it.
- **The test split does not look like production, so the results are an upper
  bound.** Studio "fresh" photos against amateur "rotten" photos invite a
  shortcut: the model can learn the photo style instead of the rot. And
  because the rot is mostly advanced, the test set cannot tell me how the
  model handles a small bruise under conveyor lighting. That is the case that
  costs money.

## Finetuning and baselines

I treated this as **image classification** with `yolo11n-cls`, finetuned from its
ImageNet checkpoint:
- **Why classification:** one image is one piece and the decision is made per
  piece, so there is nothing to locate.
- **Why the nano model:** it is enough for six visually distinct classes, and
  it classifies an image in about 3 ms on a CPU. That matters on a line that
  moves two pieces a second.

**Where and how I trained it:**
- **Machine:** the Claude Code cloud sandbox: a 4-core Intel Xeon CPU, no GPU,
  PyTorch 2.14 (CPU build), Ultralytics 8.4.183.
- **Settings:** 30 epochs at 224 px, batch 64, AdamW at a learning rate of
  0.001 (chosen automatically by Ultralytics), early-stopping patience 10,
  seed 0. It took 65 minutes.
- **Best epoch:** validation accuracy peaked at epoch 20 (99.8%), and that
  epoch is the `weights/best.pt` I committed.
- **Reproducing it:** `uv run python main.py train` repeats everything end to
  end, from download to the files the dashboard reads.

Results on the same test split (1,242 images, 234 independent photos):

| model | accuracy | macro precision | macro recall | macro F1 | rotten-vs-fresh ROC AUC |
|---|---|---|---|---|---|
| **finetuned yolo11n-cls** | **0.986** | **0.985** | **0.985** | **0.985** | **0.999** |
| COCO `yolo11n`, zero-shot | 0.473 | 0.323 | 0.519 | 0.391 | 0.647 |
| majority class (`rottenapples`) | 0.221 | 0.037 | 0.167 | 0.060 | 0.500 |

Ultralytics' own validator gives the same 0.986 on the test split, so my
evaluation code and theirs agree.

**How I built the baselines.**

- **COCO zero-shot.** The pretrained COCO detector knows *apple*, *banana* and
  *orange*, but has no idea what rot is. I map its most confident fruit
  detection to the *fresh* class of that fruit; if it finds no fruit, I fall
  back to the majority class.
  - It is good at naming fresh fruit: 82–97% recall on the fresh classes.
  - It never calls a banana or an orange rotten.
  - Its AUC of 0.65 comes entirely from rotten fruit often being too far gone
    to be recognised as fruit at all, which triggers the fallback.
  - The lesson: telling *which fruit* it is comes almost for free. Telling
    *fresh from rotten* is what the finetuning adds.
- **Majority class.** It calls everything a rotten apple. As a business
  decision, that means binning every piece on the line.

**Was the gap what I expected?** The model beats both baselines by a wide
margin. But 98.6% is higher than I would trust a nano model to reach on a real
line. Even with the leak removed, this dataset is easy: badly rotten fruit next
to studio shots of perfect fruit. I read 98.6% as a ceiling, not a forecast.

## Per-class evaluation and failure modes

I treat **rotten** as the positive class: `rottenapples`, `rottenbanana` and
`rottenoranges` together. The model's rotten score is the sum of those three
probabilities.

| class | test images | precision | recall | F1 |
|---|---|---|---|---|
| freshapples | 185 | 0.989 | 1.000 | 0.995 |
| freshbanana | 174 | 0.983 | 1.000 | 0.991 |
| **freshoranges** | 165 | 0.963 | **0.952** | 0.957 |
| rottenapples | 275 | 1.000 | 0.996 | 0.998 |
| rottenbanana | 254 | 0.984 | 0.992 | 0.988 |
| **rottenoranges** | 189 | 0.989 | **0.968** | 0.979 |

Confusion matrix on test (rows are the true class, columns the prediction):

| | fresh apple | fresh banana | fresh orange | rotten apple | rotten banana | rotten orange |
|---|---|---|---|---|---|---|
| fresh apple | 185 | 0 | 0 | 0 | 0 | 0 |
| fresh banana | 0 | 174 | 0 | 0 | 0 | 0 |
| fresh orange | 2 | 1 | 157 | 0 | 3 | 2 |
| rotten apple | 0 | 0 | 0 | 274 | 1 | 0 |
| rotten banana | 0 | 2 | 0 | 0 | 252 | 0 |
| rotten orange | 0 | 0 | 6 | 0 | 0 | 183 |

(Ultralytics' version of the same matrix is in
`artifacts/plots/test_confusion_matrix.png`.)

**What carries the headline and what drags it.**
- **Apples and bananas carry the 98.6%.** Fresh apples and fresh bananas are
  never misclassified, and every rotten apple is called rotten.
- **Oranges drag it down.** Fresh oranges are the weakest class (95.2%
  recall). The most expensive confusion in the table is rotten orange called
  fresh orange: 6 images, each of which would ship a rotten piece.

**The errors are not spread out; they come in clumps.** All 17 misclassified
test images come from just **6 photos**, and the 7 decisions that actually cost
money come from **4 photos**. So the per-class numbers above rest on a handful
of independent mistakes; one more bad photo would move a recall by one to
three points. I looked at each of the four, and each tells a different story
(they are listed in `artifacts/errors_test.csv` and in the dashboard):

1. **A shrivelled, dried-out orange with no mould gets shipped** (4 of its 7
   copies pass; the other 3 go to review). The model has learned "rotten
   orange = white or green mould" and does not recognise drying out as rot.
   This is the failure I would worry about most on a real line.
2. **A black banana photographed on a bright yellow studio backdrop gets
   shipped** (one copy). The studio look reads as "fresh". This is the
   photo-style shortcut I was worried about in the dataset section, showing
   up in practice.
3. **A heap of fresh oranges, rotated with black corners, gets binned.** It
   is not a single piece of fruit, so it is outside what the line would show.
4. **A fresh orange with a dried brown leaf attached gets binned.** The leaf
   reads as rot.

The other confusions, such as rotten apple → rotten banana or fresh orange →
fresh apple, cost nothing. The decision only cares about fresh against rotten.

## Cost matrix and operating point

The costs per piece, from the business section:

| actual \ decision | ship (pass) | grading table (review) | bin (fail) |
|---|---|---|---|
| rotten | **€4.00** (FN) | €0.015 + 3% × €4.00 (grader misses it) | €0 (TP) |
| fresh | €0 (TN) | €0.015 + 2% × €0.30 (grader bins it) | **€0.30** (FP) |

**Base rate: 4% of incoming pieces are rotten.** The test split is 57.8%
rotten, so reading costs straight off the test counts would make catching rot
look about 14 times more valuable than it is. I compute the model's catch rate
on rotten fruit and its false-alarm rate on fresh fruit separately, then
re-weight them to the 4% line.

**How I picked the threshold.**
- I swept the threshold on the rotten score from 0 to 1 in steps of 0.01, on
  the **validation** split, and computed the expected cost per 1,000 pieces at
  every step.
- The cheapest point was **0.70**. I then report what 0.70 costs on the
  **test** split, which played no part in choosing it.

![Cost against threshold](artifacts/plots/cost_vs_threshold.png)

The same curve is live in the dashboard, where you can move the base rate and
the cost of a missed rotten piece and watch it change. The sweep data is in
`artifacts/cost_sweep.csv`.

**Theory agrees with the sweep.** The break-even chance of being rotten is
7%. The model was trained on 57.5% rotten images, not 4%, so its scores
are shifted. Correcting the 7% for that difference in odds puts the cut on
the model's score at 0.71. The sweep landed at 0.70.

**Reading the curve honestly.**
- **The middle is flat.** Every threshold between about 0.16 and 0.95 costs
  between €2 and €8 per 1,000 pieces on test.
- **The edges explode.** Below 0.05 the model bins most of the fresh fruit;
  at 1.0 a quarter of the rotten fruit ships.
- **Inside the flat stretch, one or two photos decide where the exact minimum
  falls.** That is why validation says 0.70 and test would say 0.95. The fair
  claim is "anything from about 0.2 to 0.9 costs roughly the same on this
  data", not "0.70 is the optimum".

At 0.70, with the model deciding alone, the test split gives:
- 98.7% of rotten pieces binned and 0.4% of fresh pieces wrongly binned;
- an expected **€3.1 per 1,000 pieces**. Bootstrapping by photo rather than by
  image (copies of one photo are not independent) puts the 95% range at
  €0.4–7.2.

**How the answer changes with my assumptions.** I re-ran the whole choice
(thresholds re-picked on validation, cost measured on test) across shipped-
rotten costs from €0.50 to €20 and base rates from 1% to 15%. The full table
is in `artifacts/sensitivity.csv`.

- **The threshold depends on the euro figures.** It stays at 0.70 as long as
  a shipped rotten piece costs less than about €6 at a 4% base rate. Above
  that, or if the rotten share climbs to 8% with a €3–4 penalty, the cheapest
  threshold drops to 0.16: the model should bin more aggressively.
- **The review band is worth it only above a certain penalty.** At a 4% base
  rate it pays for itself once a shipped rotten piece costs more than about
  €2. Below that, the model alone is cheaper, because the reviews cost more
  than the few mistakes they catch.
- **The model beats people in every scenario.** Whatever the costs and base
  rate in that grid, the model (alone or with review) costs less than a third
  of having a person check every piece, and a tiny fraction of shipping
  everything.

**The assumption that would change my answer most is not a euro figure. It is
how often the model misses rot on real conveyor fruit.**
- The model beats a person checking every piece (€25.6 per 1,000) as long as,
  roughly, 160 × miss rate + 288 × false-alarm rate stays below 25.6.
- With false alarms around 0.4%, that means it can afford to miss up to about
  **15% of rotten pieces** in production.
- On this test it misses 0.7% (with review) to 1.3% (alone). That is a large
  margin, but this test set contains very little of the borderline rot that a
  real line sees.

## What a person still has to check

I chose the two thresholds together on validation. I searched every pair (lo,
hi) at steps of 0.01, keeping only pairs whose review load fits the graders'
capacity, and took the cheapest:

- **pass** if the rotten score is below **0.16**;
- **review** if it is between **0.16 and 0.70**;
- **fail** (bin) if it is **0.70 or above**.

The capacity limit never came into play: the search without it picks the same
band.

On the test split, re-weighted to 4% rotten:
- **2.0% of pieces go to the grading table.** They are mostly fresh oranges
  and fresh apples the model is unsure about, plus a handful of rotten
  oranges.
- 0.7% of rotten pieces still ship; 0.4% of fresh pieces are binned.

**The review load:** 2.0% × 57,600 = **1,174 pieces a day**. At 3 seconds each
that is **59 minutes a day**, or **0.14** of one grader's 7 productive hours.

**Is that realistic?** Yes, comfortably.
- **The work fits the line.** Diverted pieces collect in a tray, and one
  grader can empty it a few times per shift; fruit does not go off in
  minutes.
- **There is plenty of headroom.** The two graders already on shift have 14
  hours of capacity between them, so the review band could grow about
  **12-fold** before staffing becomes a problem.
- **That headroom will be needed.** On real conveyor images the band will
  grow, because borderline fruit is far more common there than in this
  dataset.

**Priced against the alternatives** (test split, 4% rotten, 57,600 pieces a day;
95% ranges from the photo-level bootstrap):

| policy | € per 1,000 pieces | € per day | graders needed | what still goes wrong |
|---|---|---|---|---|
| Do nothing (ship everything) | 160.0 | 9,216 | 0 | every rotten piece ships |
| A person checks every piece | 25.6 | 1,472 | **6.9** (48 h of checking a day) | 3% of rotten still ships; needs about 5 more graders than the line has |
| Model alone (threshold 0.70) | 3.1 (0.4–7.2) | 179 | 0 | 1.3% of rotten ships and nobody looks at it |
| **Model + review (0.16 / 0.70)** | **2.7 (0.4–5.5)** | **154** | **0.14** | 0.7% of rotten ships |

**My recommendation is to run the model with the review band.**
- **It is the cheapest policy, though not by a margin I can prove.** Its edge
  over the model alone, about €25 a day, sits inside the uncertainty: the two
  ranges overlap.
- **The real reason I keep the band is risk.** It costs about one
  grader-hour a day. It keeps a person looking at exactly the cases the model
  is unsure of. And it gives the line a cheap buffer for the worse
  performance I expect on real conveyor images.
- **Either model policy beats both alternatives by far.** Each costs about a
  tenth of checking every piece by hand, an option the line could not even
  staff with two graders, and about 2% of shipping everything unchecked.

## Limitations & next steps

1. **The images are not from a conveyor.** These are web and stock photos,
   with studio "fresh" shots, amateur "rotten" shots and mostly advanced rot,
   so I expect real recall to be lower than 98.7%.
   - Before going live, I would photograph a day or two of real line output
     and label it with the graders' own decisions.
   - Then I would re-measure the miss rate. The whole cost case rests on it
     staying below about 15%.
2. **Few independent photos.** The test set is 234 photos, about 39 per class.
   Per-class recall carries about ±10 points of uncertainty, and every costly
   error so far comes from just 4 photos. More distinct photos per class would
   tighten that. I already bootstrap the costs by photo rather than by image,
   for exactly this reason.
3. **The threshold is underdetermined.** The cost curve is flat from about 0.2
   to 0.9, because the model makes too few mistakes on this data to pin the
   minimum down. On real line images, where mistakes will be more common, I
   would re-run the sweep and recalibrate the scores to the line's real rotten
   share.
4. **The published splits leaked.** I fixed this by re-splitting on the
   source photo, recovered from the filenames. A pixel-hash check found only 5
   test images (0.4%) that still look like a training image. Grouping by image
   similarity instead of filename would close that last gap.
5. **Specific blind spots:** rot without mould (dried-out oranges), dark fruit
   on studio backgrounds, and leaves. I would add examples of each to
   training, and audit a sample of passed fruit every week to catch new
   failure modes early.
6. **The euro figures and the 4% base rate are my assumptions.** I would
   measure the real rotten share and the retailer's actual charge per claim,
   then re-run the sweep; that is a one-line change in `config.py`.

## Generative AI use disclosure

I used Claude Code (Anthropic) as an assistant throughout this assignment,
with my professor's permission for it to also draft the report text.

- **Code.** Claude Code wrote all of it: the training pipeline (`train.py`),
  the shared metrics and cost logic (`evaluation.py`), the settings file
  (`config.py`) and the Gradio dashboard (`app.py`). It also ran the training
  and the self-check in its cloud environment.
- **Dataset.** It compared the candidate datasets through the Roboflow API and
  recommended the one I used. It also found the leakage between the published
  splits and wrote the re-split by source photo.
- **Costs.** It proposed the cost figures, the base rate and the staffing
  assumptions, which I reviewed and accepted as reasonable for a mid-size
  packing line.
- **Report.** It drafted the text of this report from the pipeline's outputs.
  I reviewed it and asked for the final version to be written in plain
  language.
- **My own decisions.** I chose the business problem (fruit grading, from the
  options it laid out). I approved the dataset, the cost assumptions, the
  thresholds the sweep selected, and the final recommendation. I take
  responsibility for every number here, and I can explain each one without
  the code in front of me.
