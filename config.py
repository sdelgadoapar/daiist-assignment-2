"""
Every assumption the pipeline and the dashboard depend on, in one place.

train.py reads this to train, evaluate and sweep thresholds; app.py reads it to
recompute costs live. Change a number here and both stay consistent. Each
figure is justified in REPORT.md, section "Cost matrix and operating point".

The problem: a fruit packing line photographs every piece of fruit on the
conveyor. Each piece is shipped (pass), diverted to a person at the grading
table (review), or knocked off into the waste/juice bin (fail).
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
WEIGHTS_DIR = ROOT / "weights"
WEIGHTS_PATH = WEIGHTS_DIR / "best.pt"
ARTIFACTS_DIR = ROOT / "artifacts"  # committed: everything the app needs
DATASET_DIR = ROOT / "datasets"     # gitignored: downloaded at train time
RUNS_DIR = ROOT / "runs"            # gitignored: Ultralytics scratch output

# --- Dataset (Roboflow Universe, used as published, not forked) -------------
ROBOFLOW_WORKSPACE = "rishwanths-workspace"
ROBOFLOW_PROJECT = "freshness-of-fruits"
ROBOFLOW_VERSION = 3  # raw images, no preprocessing or augmentation
ROBOFLOW_FORMAT = "folder"  # classification export: <split>/<class>/<image>

# The published splits share source photos: most valid/test images are a
# rotated, flipped or noised copy of a training photo. train.py therefore
# re-splits the downloaded images by source photo (every copy of a photo in
# one split), per class, with these shares. The Roboflow dataset is untouched.
SPLIT_BY_SOURCE_PHOTO = True
GROUPED_SPLIT_SHARES = {"train": 0.70, "valid": 0.15, "test": 0.15}
# Filename prefixes the original dataset's own augmentation added.
AUGMENTATION_PREFIX = r"^(rotated_by_-?\d+_|vertical_flip_|horizontal_flip_|translation_|saltandpepper_)+"

# A class is "rotten" (the positive class: the one we must not ship) when its
# name contains one of these words; the fruit is the name with the freshness
# word removed. Override either per class name if the dataset's names are odd.
ROTTEN_WORDS = ("rotten", "spoiled", "bad", "stale")
FRESH_WORDS = ("fresh", "good")
CLASS_OVERRIDES: dict[str, tuple[str, bool]] = {}  # name -> (fruit, is_rotten)

# COCO classes the zero-shot detector can map onto a fruit. COCO knows what
# an apple, a banana and an orange are; it has no notion of rotten.
COCO_TO_FRUIT = {"apple": "apple", "banana": "banana", "orange": "orange"}

# --- Training -----------------------------------------------------------------
BASE_MODEL = "yolo11n-cls.pt"  # finetuned from this ImageNet checkpoint
COCO_BASELINE = "yolo11n.pt"   # COCO detector, used zero-shot
EPOCHS = 30
IMGSZ = 224
BATCH = 64
PATIENCE = 10
SEED = 0

# --- Costs, in euros per piece of fruit -----------------------------------
# Outcome of the final decision. Justified in REPORT.md.
COST_FN = 4.00   # rotten piece shipped: customer refund/complaint on the pack
COST_FP = 0.30   # fresh piece discarded: its sale value, lost
COST_TP = 0.0    # rotten piece discarded: it was worth nothing before the
                 # camera saw it, so the decision itself costs nothing
COST_TN = 0.0    # fresh piece shipped: the business as usual

# --- Human review at the grading table ------------------------------------
REVIEW_SECONDS = 3            # a grader picks up, turns and judges one piece
REVIEWER_HOURLY_COST = 18.0   # fully loaded cost of a line worker
COST_REVIEW = REVIEW_SECONDS / 3600 * REVIEWER_HOURLY_COST
REVIEWER_SENSITIVITY = 0.97   # share of rotten pieces a grader catches
REVIEWER_SPECIFICITY = 0.98   # share of fresh pieces a grader lets through
REVIEWERS_AVAILABLE = 2       # graders who can stand at the review table
PRODUCTIVE_HOURS_PER_REVIEWER = 7.0  # per shift, after breaks

# --- Operations -----------------------------------------------------------
# One line, 2 pieces per second, one 8-hour shift.
PIECES_PER_SECOND = 2
SHIFT_HOURS = 8
DAILY_VOLUME = int(PIECES_PER_SECOND * 3600 * SHIFT_HOURS)

# Share of incoming pieces that are really rotten. The dataset is roughly
# balanced by design, so this is NOT read from the test split.
BASE_RATE = 0.04

# --- Threshold grid ---------------------------------------------------------
THRESHOLD_STEP = 0.01
