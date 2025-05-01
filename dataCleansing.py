from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
import pandas as pd, re, pathlib

RAW_CSV   = "data/Twitter_Data.csv"
MODEL_NAME = "ProsusAI/finbert"
OUT_DIR    = "data/finbert_tweets"

# 1️⃣  Load CSV and rename columns
df = pd.read_csv(RAW_CSV)
df = df.rename(columns={"clean_text": "text", "category": "label"})

# 2️⃣  Convert label to int, map to FinBERT IDs
df["label"] = pd.to_numeric(df["label"], downcast="integer")

label_map = {-1: 0,   # bearish
              0: 1,   # neutral
              1: 2}   # bullish

df["label"] = df["label"].map(label_map)
df = df.dropna(subset=["label"])

print("Label counts after mapping:\n", df["label"].value_counts())

# 3️⃣  Clean text
df["text"] = (
    df["text"]
      .astype(str)
      .apply(lambda t: re.sub(r"http\S+|@\w+|#[A-Za-z0-9_]+|\$[A-Za-z]+", "", t))
      .str.lower()
      .str.strip()
)

# 4️⃣  Train/validation split
train_df, val_df = train_test_split(
    df, test_size=0.15, stratify=df["label"], random_state=42
)

# 5️⃣  Create 🤗 Datasets from DataFrames
dset = DatasetDict({
    "train":      Dataset.from_pandas(train_df.reset_index(drop=True)),
    "validation": Dataset.from_pandas(val_df.reset_index(drop=True)),
})

# 6️⃣  Tokenise
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

def tok(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=96,
    )

dset = dset.map(tok, batched=True).remove_columns(["text"])

# 7️⃣  Save to disk
pathlib.Path(OUT_DIR).parent.mkdir(parents=True, exist_ok=True)
dset.save_to_disk(OUT_DIR)
print(f"\n✅  Pre-tokenised dataset written to {OUT_DIR}")
