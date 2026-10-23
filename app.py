"""Step 4 - Demo app.  Run:  streamlit run app.py
Paste a Reddit-style post -> the model says 'depression-like' or 'suicide-risk-like',
shows its confidence, and highlights the words that pushed the prediction.
NOT a diagnostic tool: research prototype only.
"""
import os
import re
import runpy
import sys
from pathlib import Path

import joblib
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import matplotlib
matplotlib.use("Agg")
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Always work from the project folder, wherever the app is started from.
PROJECT_DIR = Path(__file__).resolve().parent
os.chdir(PROJECT_DIR)
sys.path.insert(0, str(PROJECT_DIR / "src"))

MODEL_PATH = "models/LogisticRegression.joblib"
if not Path(MODEL_PATH).exists():
    # First run: train the model automatically (takes about 10 seconds).
    runpy.run_path("src/06_train_app_model.py", run_name="__main__")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

def clean(text: str) -> str:
    """Light cleaning like the training data: lowercase, letters only, stop-words removed."""
    words = re.sub(r"[^a-z\s]", " ", text.lower()).split()
    return " ".join(w for w in words if w not in ENGLISH_STOP_WORDS)

def explain(pipe, text, top_k=8):
    """Words with the largest contribution (tf-idf value x model weight) for this post."""
    tfidf, clf = pipe.named_steps["tfidf"], pipe.named_steps["clf"]
    x = tfidf.transform([clean(text)])
    words = np.array(tfidf.get_feature_names_out())
    contrib = x.multiply(clf.coef_[0]).toarray()[0]
    order = np.argsort(np.abs(contrib))[::-1][:top_k]
    return [(words[i], float(contrib[i])) for i in order if contrib[i] != 0]

def lime_explanation(pipe, text, num_features=8, num_samples=600):
    """LIME: hide random words from the post many times, see how the prediction changes,
    and fit a small interpretable model to learn which words mattered locally."""
    from lime.lime_text import LimeTextExplainer
    explainer = LimeTextExplainer(class_names=["depression", "suicide"], random_state=42)
    predict_fn = lambda texts: pipe.predict_proba([clean(t) for t in texts])
    return explainer.explain_instance(text, predict_fn, num_features=num_features, num_samples=num_samples)

st.set_page_config(page_title="Depression vs Suicide-risk text classifier", page_icon="🧠")
st.title("Depression vs suicide-risk text classifier")
st.caption("Research prototype trained on the SDCNL Reddit dataset. It is not a diagnosis.")

pipe = load_model()
examples = {
    "(write your own)": "",
    "Example 1": "I have been feeling empty for months and I cannot enjoy anything anymore. Work is exhausting and I barely sleep.",
    "Example 2": "I do not see the point anymore. I keep thinking about ending it all and I have a plan.",
    "Example 3": "Everything feels heavy lately, my friends say I should talk to someone about how low I feel.",
}
choice = st.selectbox("Try an example or write your own:", list(examples))
text = st.text_area("Post text", value=examples[choice], height=160)

if st.button("Analyse") and text.strip():
    prob = float(pipe.predict_proba([clean(text)])[0, 1])
    label = "Suicide-risk-like" if prob >= 0.5 else "Depression-like"
    st.subheader(f"Prediction: {label}")
    st.progress(prob if prob >= 0.5 else 1 - prob, text=f"Confidence: {max(prob, 1 - prob):.0%}")
    tab_lime, tab_weights = st.tabs(["LIME explanation", "Model weights"])
    with tab_lime:
        try:
            with st.spinner("Running LIME (about 5 seconds)..."):
                exp = lime_explanation(pipe, text)
            st.caption("Green/orange bars: words that pushed the prediction toward 'suicide' (right) or 'depression' (left).")
            fig = exp.as_pyplot_figure(label=1)
            fig.set_size_inches(6, 3.2)
            st.pyplot(fig)
            components.html(exp.as_html(), height=340, scrolling=True)
        except ImportError:
            st.error("LIME is not installed. Run:  pip install lime")
    with tab_weights:
        st.write("**Words that influenced this prediction** (red = toward suicide-risk, blue = toward depression):")
        for w, c in explain(pipe, text):
            colour = "#C44E52" if c > 0 else "#4C72B0"
            st.markdown(f"<span style='color:{colour}'>**{w}**</span> ({c:+.2f})", unsafe_allow_html=True)
    if prob >= 0.5:
        st.warning("If you or someone you know is in crisis, please contact a local helpline or emergency services.")
    st.caption("Models make mistakes in both directions; never use this to make decisions about a real person.")
