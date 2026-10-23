"""Step 2 - Train and compare classical models (TF-IDF features).
Run:  python src/02_train_classical.py
Models: Logistic Regression, Linear SVM, Random Forest.
Saves: metrics table, test-set predictions (needed later for McNemar's test),
and the best model (needed by the app).
"""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import os
for _d in ('models','outputs/figures'):
    os.makedirs(_d, exist_ok=True)  # make sure output folders exist
from common import load_data, SEED

train, val, test = load_data()

def tfidf():
    # unigrams + bigrams; ignore very rare/very common words
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True)

models = {
    "LogisticRegression": (
        Pipeline([("tfidf", tfidf()), ("clf", LogisticRegression(max_iter=2000, random_state=SEED))]),
        {"clf__C": [0.5, 1, 5, 10]},
    ),
    "LinearSVM": (
        Pipeline([("tfidf", tfidf()), ("clf", CalibratedClassifierCV(LinearSVC(random_state=SEED), cv=3))]),
        {"clf__estimator__C": [0.1, 0.5, 1]},
    ),
    "RandomForest": (
        Pipeline([("tfidf", tfidf()), ("clf", RandomForestClassifier(random_state=SEED, n_jobs=-1))]),
        {"clf__n_estimators": [200, 400], "clf__max_depth": [None, 40]},
    ),
}

# tune on train+val with 5-fold CV (the validation set joins the CV pool)
pool = pd.concat([train, val]).reset_index(drop=True)
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
rows, preds = [], {}
for name, (pipe, grid) in models.items():
    search = GridSearchCV(pipe, grid, cv=cv, scoring="f1", n_jobs=-1).fit(pool["text"], pool["label"])
    best = search.best_estimator_
    p = best.predict(test["text"]); prob = best.predict_proba(test["text"])[:, 1]
    rows.append({
        "model": name, "best_params": json.dumps(search.best_params_), "cv_f1": round(search.best_score_, 4),
        "accuracy": accuracy_score(test.label, p), "precision": precision_score(test.label, p),
        "recall": recall_score(test.label, p), "f1": f1_score(test.label, p),
        "roc_auc": roc_auc_score(test.label, prob),
    })
    preds[name] = p
    preds[name + "_prob"] = prob
    joblib.dump(best, f"models/{name}.joblib")
    print(f"done {name}: test acc={rows[-1]['accuracy']:.4f} f1={rows[-1]['f1']:.4f}")

res = pd.DataFrame(rows).round(4)
res.to_csv("outputs/classical_results.csv", index=False)
pd.DataFrame(preds).assign(y_true=test.label.values).to_csv("outputs/classical_test_predictions.csv", index=False)
print(res.drop(columns="best_params").to_string(index=False))
