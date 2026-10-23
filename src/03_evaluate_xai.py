"""Step 3 - Compare models rigorously + explain the best one.
Run:  python src/03_evaluate_xai.py   (after 02_train_classical.py)
Does:
  1. Bootstrap 95% confidence intervals for F1 (how much can the score move by luck?)
  2. McNemar's test for every pair of models (is one REALLY better than another?)
  3. Confusion matrices + ROC curves
  4. Error analysis (which posts are misclassified, and which error is worse?)
  5. SHAP explanation of the Logistic Regression model (which words drive predictions?)
If outputs/distilbert_test_predictions.csv exists (from the Colab notebook) it is
included automatically as a 4th model.
"""
import itertools, os
import joblib, numpy as np, pandas as pd, shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, roc_curve, roc_auc_score, ConfusionMatrixDisplay
from statsmodels.stats.contingency_tables import mcnemar
import os
for _d in ('models','outputs/figures'):
    os.makedirs(_d, exist_ok=True)  # make sure output folders exist
from common import load_data, LABELS, SEED

train, val, test = load_data()
y = test["label"].values
preds = pd.read_csv("outputs/classical_test_predictions.csv")
assert (preds["y_true"].values == y).all(), "prediction file does not match test set order"
names = [c for c in preds.columns if not c.endswith("_prob") and c != "y_true"]
P = {n: preds[n].values for n in names}
PROB = {n: preds[n + "_prob"].values for n in names}

# optional transformer results from Colab
if os.path.exists("outputs/distilbert_test_predictions.csv"):
    d = pd.read_csv("outputs/distilbert_test_predictions.csv")
    assert (d["y_true"].values == y).all()
    P["DistilBERT"], PROB["DistilBERT"] = d["DistilBERT"].values, d["DistilBERT_prob"].values
    names.append("DistilBERT")

# ---- 1. bootstrap confidence intervals -------------------------------------
rng = np.random.default_rng(SEED)
idx = [rng.integers(0, len(y), len(y)) for _ in range(1000)]  # 1000 resamples of the test set
rows = []
for n in names:
    f1s = [f1_score(y[i], P[n][i]) for i in idx]
    lo, hi = np.percentile(f1s, [2.5, 97.5])
    rows.append({"model": n, "accuracy": accuracy_score(y, P[n]), "f1": f1_score(y, P[n]),
                 "f1_ci_low": lo, "f1_ci_high": hi, "roc_auc": roc_auc_score(y, PROB[n])})
summary = pd.DataFrame(rows).round(4)
summary.to_csv("outputs/final_results_with_ci.csv", index=False)
print(summary.to_string(index=False))

# ---- 2. McNemar's test for each pair ---------------------------------------
mc = []
for a, b in itertools.combinations(names, 2):
    ca, cb = P[a] == y, P[b] == y
    table = [[np.sum(ca & cb), np.sum(ca & ~cb)], [np.sum(~ca & cb), np.sum(~ca & ~cb)]]
    p = mcnemar(table, exact=True).pvalue
    mc.append({"model_A": a, "model_B": b, "A_right_B_wrong": table[0][1],
               "A_wrong_B_right": table[1][0], "p_value": round(p, 4),
               "significant_at_0.05": p < 0.05})
mc = pd.DataFrame(mc); mc.to_csv("outputs/mcnemar_results.csv", index=False)
print("\nMcNemar's test:\n", mc.to_string(index=False))

# ---- 3. confusion matrices and ROC curves ----------------------------------
fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 3.6))
for ax, n in zip(np.atleast_1d(axes), names):
    ConfusionMatrixDisplay.from_predictions(y, P[n], display_labels=list(LABELS.values()),
                                            ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(n)
plt.tight_layout(); plt.savefig("outputs/figures/confusion_matrices.png", dpi=200); plt.close()

plt.figure(figsize=(5, 4))
for n in names:
    fpr, tpr, _ = roc_curve(y, PROB[n]); plt.plot(fpr, tpr, label=f"{n} (AUC {roc_auc_score(y, PROB[n]):.3f})")
plt.plot([0, 1], [0, 1], "k--", lw=.8); plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
plt.legend(fontsize=8); plt.title("ROC curves (test set)")
plt.tight_layout(); plt.savefig("outputs/figures/roc_curves.png", dpi=200); plt.close()

# ---- 4. error analysis (Logistic Regression) -------------------------------
lr = joblib.load("models/LogisticRegression.joblib")
err = test.assign(pred=P["LogisticRegression"], prob_suicide=PROB["LogisticRegression"])
err = err[err.pred != err.label].copy()
err["error_type"] = np.where(err.label == 1, "missed suicide post (false negative)", "false alarm (false positive)")
err["confidence"] = np.where(err.pred == 1, err.prob_suicide, 1 - err.prob_suicide)
err["length_words"] = err.raw_text.str.split().str.len()
err.sort_values("confidence", ascending=False)[
    ["error_type", "confidence", "length_words", "raw_text"]].to_csv("outputs/error_analysis_LR.csv", index=False)
print("\nError types:\n", err.error_type.value_counts().to_string())
print("Mean length (words)  errors:", round(err.length_words.mean(), 1),
      " all test posts:", round(test.raw_text.str.split().str.len().mean(), 1))

# ---- 5. SHAP explanation of Logistic Regression ----------------------------
tfidf, clf = lr.named_steps["tfidf"], lr.named_steps["clf"]
words = np.array(tfidf.get_feature_names_out())
Xtr = tfidf.transform(train["text"]).toarray()
Xte = tfidf.transform(test["text"]).toarray()
explainer = shap.LinearExplainer(clf, Xtr)       # exact SHAP values for a linear model
sv = explainer.shap_values(Xte)                  # shape: (posts, words); + pushes toward 'suicide'

mean_abs = np.abs(sv).mean(0); top = np.argsort(mean_abs)[::-1][:20][::-1]
direction = np.sign(clf.coef_[0][top])
plt.figure(figsize=(6, 5))
plt.barh(words[top], mean_abs[top], color=np.where(direction > 0, "#C44E52", "#4C72B0"))
plt.xlabel("mean |SHAP value|"); plt.title("Top words (red = pushes to 'suicide', blue = 'depression')")
plt.tight_layout(); plt.savefig("outputs/figures/shap_global.png", dpi=200); plt.close()

def local_plot(i, fname):
    """Bar chart of the words that mattered most for ONE post."""
    contrib = sv[i]; top_i = np.argsort(np.abs(contrib))[::-1][:10][::-1]
    plt.figure(figsize=(6, 3.5))
    plt.barh(words[top_i], contrib[top_i], color=np.where(contrib[top_i] > 0, "#C44E52", "#4C72B0"))
    plt.axvline(0, color="k", lw=.8); plt.xlabel("SHAP value (right = toward 'suicide')")
    plt.title(f"True: {LABELS[y[i]]} | Predicted: {LABELS[P['LogisticRegression'][i]]}")
    plt.tight_layout(); plt.savefig(fname, dpi=200); plt.close()

ok = np.where(P["LogisticRegression"] == y)[0]
local_plot(ok[y[ok] == 1][0], "outputs/figures/shap_local_suicide_example.png")
local_plot(ok[y[ok] == 0][0], "outputs/figures/shap_local_depression_example.png")
pd.DataFrame({"word": words[np.argsort(mean_abs)[::-1][:30]],
              "mean_abs_shap": np.sort(mean_abs)[::-1][:30]}).to_csv("outputs/shap_top_words.csv", index=False)
print("\nTop 15 words by SHAP:", list(words[np.argsort(mean_abs)[::-1][:15]]))
print("Figures saved in outputs/figures/")
