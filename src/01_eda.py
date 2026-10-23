"""Step 1 - Exploratory Data Analysis (EDA).
Run from the project root:  python src/01_eda.py
Produces figures in outputs/figures/ that go straight into the report.
"""
import re
from collections import Counter
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # draw to files, no window needed
import matplotlib.pyplot as plt
import os
for _d in ('models','outputs/figures'):
    os.makedirs(_d, exist_ok=True)  # make sure output folders exist
from common import load_data, LABELS

train, val, test = load_data()
raw = pd.concat([train, val, test])

# 1. Dataset statistics (goes in the Dataset Description section)
print("Samples  train/val/test:", len(train), len(val), len(test))
for name, d in [("train", train), ("val", val), ("test", test)]:
    print(name, d["label"].map(LABELS).value_counts().to_dict())

# 2. Class balance
counts = train["label"].map(LABELS).value_counts()
plt.figure(figsize=(4, 3)); counts.plot(kind="bar", color=["#4C72B0", "#C44E52"])
plt.title("Class balance (training set)"); plt.ylabel("posts"); plt.xticks(rotation=0)
plt.tight_layout(); plt.savefig("outputs/figures/class_balance.png", dpi=200); plt.close()

# 3. Post length: before vs after cleaning (preprocessing effect)
raw_len = train["raw_text"].str.split().str.len()
clean_len = train["text"].str.split().str.len()
print("Mean words raw / clean:", round(raw_len.mean(), 1), round(clean_len.mean(), 1))
plt.figure(figsize=(6, 3))
plt.hist(raw_len.clip(upper=800), bins=40, alpha=.6, label="raw")
plt.hist(clean_len.clip(upper=800), bins=40, alpha=.6, label="cleaned")
plt.xlabel("words per post (capped at 800)"); plt.ylabel("posts"); plt.legend()
plt.title("Post length before vs after preprocessing")
plt.tight_layout(); plt.savefig("outputs/figures/length_before_after.png", dpi=200); plt.close()

# 4. Length by class
print(train.assign(n=clean_len).groupby("label")["n"].describe().round(1))

# 5. Most frequent words per class
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax, (lab, name) in zip(axes, LABELS.items()):
    words = Counter(" ".join(train.loc[train.label == lab, "text"]).split())
    top = words.most_common(15)[::-1]
    ax.barh([w for w, _ in top], [c for _, c in top]); ax.set_title(f"Top words: {name}")
plt.tight_layout(); plt.savefig("outputs/figures/top_words.png", dpi=200); plt.close()
print("Figures saved in outputs/figures/")
