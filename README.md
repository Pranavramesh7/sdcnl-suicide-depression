# Suicide vs Depression text classification (SDCNL) - PRG4206
Research prototype - NOT a diagnostic tool.

## Quick start: run the demo app (about 3 minutes)
    pip install -r requirements.txt
    python src/06_train_app_model.py     # makes the model file (about 10 seconds)
    streamlit run app.py                 # opens the demo in your browser

## Full pipeline (from this folder)
1. `python src/01_eda.py`               - EDA figures -> outputs/figures/
2. `python src/02_train_classical.py`   - Logistic Regression / Linear SVM / Random Forest (TF-IDF)
3. `colab/05_distilbert_colab.ipynb`    - fine-tune DistilBERT on a Colab T4 GPU -> outputs/distilbert_test_predictions.csv
4. `python src/03_evaluate_xai.py`      - bootstrap CIs, McNemar's test, ROC/confusion, error analysis, SHAP
(`colab/00_run_everything_colab.ipynb` runs steps 1-4 in Colab without any local setup.)

## Data
SDCNL (Haque et al., 2021): https://github.com/ayaanzhaque/SDCNL  (official train/test split kept).
Author (username) columns are never used. Random seed 42 everywhere.
