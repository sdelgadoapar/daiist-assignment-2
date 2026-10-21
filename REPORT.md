# Assignment 2 Report

- **Name**: Sebastián Delgado-Aparicio
- **Student ID**:
- **Email**: sdelgadoapar.ieu2022@student.ie.edu
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
throwing away a good one. So a piece should be thrown out as soon as the
chance it is rotten passes 0.30 / (0.30 + 4.00) ≈ **7%**, not 50%. The model's
score is not that chance directly: it was trained on 57.5% rotten images, not
4%. Converting the 7% to the model's scale gives a cut at about **0.71**
(see Cost matrix section). The threshold is chosen by sweeping cost, not by
accuracy.
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

> DRAFT: to be reviewed and adjusted. Numbers from `artifacts/summary.md`.

**Model and task.** Image classification with `yolo11n-cls`, finetuned from its
ImageNet checkpoint.
- Classification fits because one image is one piece and the decision is per
  piece. Nothing needs to be located.
- The nano model is enough for six visually distinct classes, and it is fast:
  about 3 ms per image on a CPU. A line handling 2 pieces per second needs
  that.

**Where and how it was trained.**
- Machine: the Claude Code cloud sandbox, 4-core Intel Xeon CPU, no GPU.
- Software: PyTorch 2.14 (CPU), Ultralytics 8.4.183.
- Settings: 30 epochs at 224 px, batch 64, optimizer chosen automatically
  (AdamW, lr 0.001), early-stopping patience 10, seed 0. Took 65 minutes.
- Validation accuracy peaked at epoch 20 (99.8%); `best.pt` is that epoch.
- `uv run python main.py train` reproduces it end to end.

**Results on the same test split** (1,242 images, 234 independent photos):

| model | accuracy | macro precision | macro recall | macro F1 | rotten-vs-fresh ROC AUC |
|---|---|---|---|---|---|
| **finetuned yolo11n-cls** | **0.986** | **0.985** | **0.985** | **0.985** | **0.999** |
| COCO `yolo11n` zero-shot | 0.473 | 0.323 | 0.519 | 0.391 | 0.647 |
| majority class (`rottenapples`) | 0.221 | 0.037 | 0.167 | 0.060 | 0.500 |

Ultralytics' own validator gives the same 0.986 top-1 on the test split.

**How the baselines were built:**
- **COCO zero-shot.** The COCO detector knows *apple*, *banana* and *orange*
  but has no notion of rot. Its top detection is mapped to the *fresh* class
  of that fruit. If it detects no fruit, it falls back to the majority class.
  - It names fresh fruit well: 82–97% recall on the fresh classes.
  - It never calls a banana or an orange rotten.
  - Its AUC of 0.65 comes only from rotten fruit often being unrecognisable
    as fruit at all, which triggers the fallback.
  - Telling *which fruit* is easy. Telling *fresh from rotten* is what the
    finetuning adds.
- **Majority class.** Calls everything a rotten apple. As a decision, that
  means discarding every piece of fruit.

**Was the gap as large as expected?** Against the baselines, yes. On its own,
98.6% is higher than a model of this size should get on a real line.
- The gap is larger than it would be had we used the published split. On that
  split, test images are copies of training photos, so even the honest
  number would have looked inflated, which is why we re-split.
- Even on clean splits the classes are easy here: severely rotten fruit
  against studio photos of fresh fruit. Treat 98.6% as a ceiling, not a
  forecast.

## Per-class evaluation and failure modes

> DRAFT: to be reviewed and adjusted.

**Positive class: rotten**, i.e. `rottenapples` + `rottenbanana` +
`rottenoranges`. The model's rotten score is the sum of those three
probabilities.

| class | test images | precision | recall | F1 |
|---|---|---|---|---|
| freshapples | 185 | 0.989 | 1.000 | 0.995 |
| freshbanana | 174 | 0.983 | 1.000 | 0.991 |
| **freshoranges** | 165 | 0.963 | **0.952** | 0.957 |
| rottenapples | 275 | 1.000 | 0.996 | 0.998 |
| rottenbanana | 254 | 0.984 | 0.992 | 0.988 |
| **rottenoranges** | 189 | 0.989 | **0.968** | 0.979 |

Confusion matrix on test (rows true, columns predicted):

| | fresh apple | fresh banana | fresh orange | rotten apple | rotten banana | rotten orange |
|---|---|---|---|---|---|---|
| fresh apple | 185 | 0 | 0 | 0 | 0 | 0 |
| fresh banana | 0 | 174 | 0 | 0 | 0 | 0 |
| fresh orange | 2 | 1 | 157 | 0 | 3 | 2 |
| rotten apple | 0 | 0 | 0 | 274 | 1 | 0 |
| rotten banana | 0 | 2 | 0 | 0 | 252 | 0 |
| rotten orange | 0 | 0 | 6 | 0 | 0 | 183 |

**What carries the headline and what drags it:**
- **Apples and bananas carry the headline.** Fresh apples and fresh bananas are
  never misclassified, and rotten apples are always called rotten.
- **Oranges drag it.** Fresh oranges are the weakest class (95.2% recall), and
  the one costly confusion that matters most is rotten orange → fresh orange
  (6 images).

**Errors come in clumps.** All 17 misclassified test images come from just
**6 photos**, and the 7 costly decisions from **4 photos**. The averages above
therefore rest on a handful of independent mistakes; one more bad photo would
move a per-class recall by 1–3 points. Each costly photo is listed in
`artifacts/errors_test.csv` and in the dashboard:

1. **A shrivelled, dehydrated rotten orange without mould is shipped** (4 of
   its 7 copies pass, the other 3 go to review). The model learned "rotten orange = white or green
   mould" and does not recognise drying out as rot.
2. **A black rotten banana on a yellow studio backdrop is shipped** (its
   salt-and-pepper copy). Studio styling reads as "fresh". This is the
   photo-style shortcut the dataset invites.
3. **A pile of many fresh oranges, rotated with black corners, is binned.**
   Not one piece of fruit, so outside what the line would show.
4. **A fresh orange with a dried brown leaf is binned.** The leaf reads as rot.

Fruit-type confusions (rotten apple → rotten banana, fresh orange → fresh
apple) cost nothing: the decision depends only on fresh or rotten.

## Cost matrix and operating point

> DRAFT: to be reviewed and adjusted. Figures live in `config.py`.

**Costs per piece**, as justified in the business section:

| actual \ decision | ship (pass) | grader (review) | discard (fail) |
|---|---|---|---|
| rotten | **€4.00** (FN) | €0.015 + 97% × €0 + 3% × €4.00 | €0 (TP) |
| fresh | €0 (TN) | €0.015 + 2% × €0.30 | **€0.30** (FP) |

**Base rate: 4% of incoming pieces are rotten.** The test split is 57.8%
rotten, so raw test counts would overstate the value of catching rot about
14-fold. Every cost is computed from the model's rates on rotten and on fresh
fruit separately, then re-weighted to 4%.

**How the threshold was chosen.**
- The model-alone threshold on the rotten score is swept from 0 to 1 in steps
  of 0.01 on **validation**. The cheapest point is **0.70**, reported on
  **test**.
- The plot is in the dashboard ("Cost against threshold"), with 0.70 marked;
  the data is in `artifacts/cost_sweep.csv`.
- Theory agrees. With a 7% break-even chance, and odds shifted from the 57.5%
  training prevalence to the 4% line prevalence, the cut on the model's score
  is 0.71.

**The curve is flat between about 0.16 and 0.95.**
- On test, every threshold in that range costs €2–8 per 1,000 pieces.
- Below 0.05 the cost explodes (most fresh fruit is binned). At 1.0 it
  explodes too (a quarter of rotten fruit ships).
- Inside the flat region, the exact point is decided by one or two photos.
  The validation optimum (0.70) and the test optimum (0.95) differ only by
  that.
- So the defensible statement is "anywhere from 0.2 to 0.9 costs about the
  same on this data", not "0.70 is optimal".

**Result at 0.70, model alone, on test:**
- 98.7% of rotten pieces discarded, 0.4% of fresh pieces discarded.
- **€3.1 per 1,000 pieces**, with a 95% range of €0.4–7.2 from bootstrapping
  test *photos* rather than images.

**Sensitivity** (`artifacts/sensitivity.csv`: thresholds re-chosen on val for
each scenario, cost measured on test):
- **Raising the FN cost or the base rate lowers the threshold.** At €6+ per
  shipped rotten piece and a 4% base rate, the cheapest model-alone threshold
  drops from 0.70 to 0.16.
- **The model always wins.** In every scenario from €0.50 to €20 per missed
  piece and 1% to 15% base rate, model-alone and model + review both cost
  under a third of "person checks everything", and far less than shipping
  everything.

**The assumption that would most change the answer is the model's miss rate
on real conveyor fruit, not any euro figure.**
- The model beats a person checking every piece (€25.6 per 1,000) as long as
  roughly 160 × miss rate + 288 × false-alarm rate < 25.6 (per 1,000 pieces).
- With false alarms near 0.4%, that means it must miss fewer than **about 15%
  of rotten pieces** in production. On this test it misses 0.7–1.3%.
- The margin is large, but this test set does not contain the borderline rot
  a real line sees.

## What a person still has to check

> DRAFT: to be reviewed and adjusted.

**Two thresholds**, chosen together on validation by searching every
(lo, hi) pair at 0.01 steps, within review capacity:
- **pass** if the rotten score is **< 0.16**;
- **review** if it is **between 0.16 and 0.70**;
- **fail** if it is **≥ 0.70**.

Review capacity (29% of pieces) never bound: the unconstrained search picks
the same band.

**On test** (re-weighted to 4% rotten):
- **2.0% of pieces go to a grader.** These are mostly fresh oranges and fresh
  apples the model is unsure about, plus a few rotten oranges.
- 0.7% of rotten pieces still ship; 0.4% of fresh pieces are binned.

**Review load:**
- 2.0% × 57,600 = **1,174 pieces a day**.
- At 3 seconds each, **59 minutes a day**: **0.14** of a grader's 7 productive
  hours.

**Is that realistic?** Yes, with room to spare.
- Diverted pieces collect in a tray, and one grader clears it a few times per
  shift: fruit does not spoil in minutes.
- The 2 graders on shift have 14 hours of capacity, so the review band could
  grow **12-fold** before staffing becomes the constraint. On a real line it
  will grow, because borderline fruit is more common there than in this
  dataset. That headroom is the main reason to keep the band rather than let
  the model decide alone.

**Priced against the alternatives** (test split, 4% rotten, 57,600 pieces/day;
95% ranges from a photo-level bootstrap):

| policy | € per 1,000 pieces | € per day | graders needed | what goes wrong |
|---|---|---|---|---|
| Do nothing (ship everything) | 160.0 | 9,216 | 0 | every rotten piece ships |
| Person checks every piece | 25.6 | 1,472 | **6.9** (48 h/day of checking) | 3% of rotten still ships; needs 5 more graders than the line has |
| Model alone (t = 0.70) | 3.1 (0.4–7.2) | 179 | 0 | 1.3% of rotten ships, nobody sees it |
| **Model + review (0.16 / 0.70)** | **2.7 (0.4–5.5)** | **154** | **0.14** | 0.7% of rotten ships |

**Recommendation.**
- Ship **model + review**. It is the cheapest policy, but its edge over the
  model alone (€25 a day) is inside the uncertainty: the two intervals
  overlap. The real reason for the band is different:
  - it costs one grader-hour a day;
  - it keeps a person looking at the cases the model is unsure about;
  - it gives the line a cheap way to absorb the worse performance expected on
    real conveyor images.
- **Both model policies cost about a tenth of having a person check
  everything**, and that option is not even feasible with 2 graders.

## Limitations & next steps

> DRAFT: to be reviewed and adjusted.

1. **The test images are not conveyor images.** These are web and stock
   photos, with studio "fresh" and amateur "rotten" shots, and mostly
   advanced rot. Production recall will be lower than 98.7%.
   - Next: photograph one or two days of real line output.
   - Label it with the graders' own decisions.
   - Re-measure the miss rate before going live. It is the number the whole
     cost case rests on (break-even ≈ 15% misses).
2. **Few independent photos.** 234 test photos, about 39 per class. Per-class
   recalls carry roughly ±10-point uncertainty, and every costly error so far
   comes from 4 photos.
   - Next: evaluate on more distinct photos, and report bootstrap intervals by
     photo (already done for cost).
3. **The threshold is underdetermined.** The cost curve is flat from about
   0.2 to 0.9 on this data, because the model makes too few errors here to
   pin it down.
   - Next: re-run the sweep on real line images, where errors are more
     common. Recalibrate the scores to the line's base rate.
4. **The published splits leak** (99% of test images are copies of training
   photos). We re-split by source photo.
   - A residual risk remains: different screenshots of the same fruit, taken
     seconds apart, cannot be detected from filenames. 54 published-test
     images had a near-identical twin in train by image hash.
   - Next: group by image similarity, not just filename.
5. **Specific blind spots:** rot without mould (dehydrated oranges), dark fruit
   on studio backgrounds, and leaves.
   - Next: add these cases to training. Audit-sample passed fruit weekly to
     catch new failure modes.
6. **The base rate and euro costs are assumptions.**
   - Next: measure the real rotten share and the actual retailer chargeback
     per claim, then re-run the sweep (one change in `config.py`).

## Generative AI use disclosure

> DRAFT: adjust to describe exactly how you used it.

I used Claude Code (Anthropic) throughout, with the professor's consent for AI
help drafting text.
- **Code:** it wrote the training pipeline, the shared evaluation and cost
  module, the Gradio dashboard, and the dataset-leakage check.
- **Text:** it drafted the sections of this report from the pipeline's
  outputs.
- **My own decisions:** the problem framing (fruit grading), and reviewing and
  adjusting the cost assumptions, thresholds, conclusions and wording.
I can explain every number and decision in this report without the code in
front of me.
