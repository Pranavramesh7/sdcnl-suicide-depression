"""Shared helpers: load SDCNL data, make the train/validation/test split.

SDCNL = Reddit posts from r/depression (label 0) and r/SuicideWatch (label 1)
(Haque et al., 2021). We keep the authors' official train/test split so our
scores can be compared with published results on the same test set.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42  # fixed seed -> anyone re-running gets the same numbers
LABELS = {0: "depression", 1: "suicide"}


def load_data():
    """Return (train, val, test) DataFrames with a single 'text' column.

    - We use the *_clean columns (stop-words removed) for classical models.
    - We deliberately DO NOT use the author columns: usernames are personal
      data and can leak the label (e.g. a name like 'wanna die badly').
    - Validation set = 15% of the official training set (stratified).
    """
    train_full = pd.read_csv("data/training-set.csv")
    test = pd.read_csv("data/testing-set.csv")

    def build(df):
        out = pd.DataFrame()
        out["text"] = (df["title_clean"].fillna("") + " " + df["selftext_clean"].fillna("")).str.strip()
        out["raw_text"] = (df["title"].fillna("") + ". " + df["selftext"].fillna("")).str.strip()
        out["label"] = df["is_suicide"].astype(int)
        return out

    train_full, test = build(train_full), build(test)
    train, val = train_test_split(
        train_full, test_size=0.15, stratify=train_full["label"], random_state=SEED
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)
