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

*Workspace / project / version on Roboflow Universe, and the link. What one
image represents, what the classes are, and why this dataset actually serves
the problem above rather than merely resembling it.*

## What the dataset's composition implies

*Instances per class, train/val/test sizes, typical object size relative to the
image. Then the interpretation: which per-class numbers will be too thin to
trust, whether the test split resembles the images you'd really see, and what
you therefore can't conclude from your results.*

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
