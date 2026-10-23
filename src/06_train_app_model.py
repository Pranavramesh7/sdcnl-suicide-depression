"""Step 6 - Train the model used by the demo app (takes about 10 seconds).
Run:  python src/06_train_app_model.py
Why this exists: a saved model file only loads reliably with the same library version it was
saved with. Running this on YOUR computer creates models/LogisticRegression.joblib that matches
your installed scikit-learn, so app.py always works.
Uses the best settings found by cross-validation in step 2 (C = 0.5).
"""
import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from common import load_data, SEED

os.makedirs("models", exist_ok=True)
train, val, test = load_data()
pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True)),
    ("clf", LogisticRegression(C=0.5, max_iter=2000, random_state=SEED)),
])
pool = pd.concat([train, val])   # train + validation
pipe.fit(pool["text"], pool["label"])
pred = pipe.predict(test["text"])
print(f"Saved models/LogisticRegression.joblib | test accuracy={accuracy_score(test.label, pred):.4f} "
      f"F1={f1_score(test.label, pred):.4f}")
joblib.dump(pipe, "models/LogisticRegression.joblib")
