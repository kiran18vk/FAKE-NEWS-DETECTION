import os
import pandas as pd
from sklearn.model_selection import train_test_split

# ============================================================
# WELFake Dataset Preparation
# Supports:
#   1. Title Only
#   2. Article Only
#   3. Title + Article
#
# Label:
#   0 = REAL
#   1 = FAKE
# ============================================================

INPUT_FILE = "WELFake_Dataset.csv"
OUTPUT_DIR = "prepared_data"

RANDOM_STATE = 42

print("=" * 70)
print("FAKE NEWS DATASET PREPARATION")
print("=" * 70)


# ============================================================
# 1. LOAD DATASET
# ============================================================

if not os.path.exists(INPUT_FILE):
    print(f"\nERROR: {INPUT_FILE} not found.")
    raise SystemExit

print("\n[1/8] Loading dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Original rows: {len(df):,}")
print(f"Original columns: {list(df.columns)}")


# ============================================================
# 2. CHECK REQUIRED COLUMNS
# ============================================================

print("\n[2/8] Checking columns...")

required_columns = ["title", "text", "label"]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    print(f"ERROR: Missing columns: {missing}")
    raise SystemExit

print("Required columns found.")


# ============================================================
# 3. KEEP USEFUL COLUMNS
# ============================================================

print("\n[3/8] Selecting useful columns...")

df = df[["title", "text", "label"]].copy()


# ============================================================
# 4. CLEAN TEXT
# ============================================================

print("\n[4/8] Cleaning text...")

df["title"] = (
    df["title"]
    .fillna("")
    .astype(str)
    .str.strip()
)

df["text"] = (
    df["text"]
    .fillna("")
    .astype(str)
    .str.strip()
)

# Remove rows where BOTH title and article are empty
df = df[
    (df["title"].str.len() > 0) |
    (df["text"].str.len() > 0)
].copy()


# ============================================================
# 5. CHECK LABELS
# ============================================================

print("\n[5/8] Checking labels...")

df["label"] = pd.to_numeric(
    df["label"],
    errors="coerce"
)

df = df[
    df["label"].isin([0, 1])
].copy()

df["label"] = df["label"].astype(int)

print("\nLabel distribution:")

for label, count in df["label"].value_counts().sort_index().items():

    name = "REAL" if label == 0 else "FAKE"

    percentage = (
        count / len(df) * 100
    )

    print(
        f"  {label} = {name}: "
        f"{count:,} ({percentage:.2f}%)"
    )


# ============================================================
# 6. REMOVE DUPLICATES
# ============================================================

print("\n[6/8] Removing duplicate articles...")

df["_title_clean"] = (
    df["title"]
    .str.lower()
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

df["_text_clean"] = (
    df["text"]
    .str.lower()
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

df = df.drop_duplicates(
    subset=["_title_clean", "_text_clean"],
    keep="first"
).copy()

df = df.drop(
    columns=["_title_clean", "_text_clean"]
)

print(f"Rows remaining: {len(df):,}")


# ============================================================
# 7. SHUFFLE
# ============================================================

print("\n[7/8] Shuffling dataset...")

df = df.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)


# ============================================================
# 8. TRAIN / VALIDATION / TEST SPLIT
# ============================================================

print("\n[8/8] Creating train/validation/test datasets...")

# FIRST split original articles
# This is important to avoid leakage.

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=df["label"]
)

validation_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=temp_df["label"]
)


# ============================================================
# SAVE ORIGINAL SPLITS
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

train_df.to_csv(
    os.path.join(OUTPUT_DIR, "train.csv"),
    index=False
)

validation_df.to_csv(
    os.path.join(OUTPUT_DIR, "validation.csv"),
    index=False
)

test_df.to_csv(
    os.path.join(OUTPUT_DIR, "test.csv"),
    index=False
)

df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "clean_fake_news.csv"
    ),
    index=False
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("DATA PREPARATION COMPLETED")
print("=" * 70)

print(f"Total articles : {len(df):,}")
print(f"Training       : {len(train_df):,}")
print(f"Validation     : {len(validation_df):,}")
print(f"Testing        : {len(test_df):,}")

print("\nLabel meaning:")
print("0 = REAL")
print("1 = FAKE")

print("\nFiles created:")
print("prepared_data/")
print("    ├── clean_fake_news.csv")
print("    ├── train.csv")
print("    ├── validation.csv")
print("    └── test.csv")