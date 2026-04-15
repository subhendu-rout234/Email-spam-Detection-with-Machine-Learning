"""
model.py — Model Training & Evaluation Module for Email Spam Detection
======================================================================
Trains Multinomial Naive Bayes and Logistic Regression classifiers,
evaluates both, selects the best, and provides save/load helpers.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from preprocess import preprocess_dataframe, build_tfidf_vectorizer

# ── Paths for persisting artefacts ───────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models")
os.makedirs(MODEL_DIR, exist_ok=True)

BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")


# ── Evaluation helper ───────────────────────────────────────────────────────
def evaluate_model(model, X_test, y_test):
    """
    Compute core classification metrics for a trained model.

    Returns
    -------
    dict
        Keys: accuracy, precision, recall, f1, confusion_matrix,
              classification_report, predictions
    """
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(
            y_test, y_pred, target_names=["Ham", "Spam"], zero_division=0
        ),
        "predictions": y_pred,
    }


# ── Main training pipeline ──────────────────────────────────────────────────
def train_and_evaluate(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    End-to-end training pipeline:
      1. Preprocess the raw DataFrame.
      2. Build TF-IDF features.
      3. Train Naive Bayes & Logistic Regression.
      4. Evaluate both and pick the best.
      5. Persist best model + vectorizer to disk.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset with text and label columns.
    test_size : float
        Fraction of data reserved for testing.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    dict
        {
            "processed_df": pd.DataFrame,
            "vectorizer": TfidfVectorizer,
            "models": {model_name: fitted_model, ...},
            "results": {model_name: metrics_dict, ...},
            "best_model_name": str,
            "best_model": estimator,
            "X_train", "X_test", "y_train", "y_test"
        }
    """
    # ── 1. Preprocess ────────────────────────────────────────────────────
    processed_df = preprocess_dataframe(df)

    if len(processed_df) < 10:
        raise ValueError(
            "Dataset too small after cleaning. Need at least 10 samples."
        )

    # ── 2. TF-IDF ────────────────────────────────────────────────────────
    vectorizer, X = build_tfidf_vectorizer(processed_df["cleaned_text"])
    y = processed_df["label"].values

    # ── 3. Train / test split ────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # ── 4. Train models ─────────────────────────────────────────────────
    models = {
        "Multinomial Naive Bayes": MultinomialNB(alpha=1.0),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, C=1.0, solver="lbfgs", random_state=random_state
        ),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        results[name] = evaluate_model(model, X_test, y_test)

    # ── 5. Select best model by accuracy ─────────────────────────────────
    best_name = max(results, key=lambda k: results[k]["accuracy"])
    best_model = models[best_name]

    # ── 6. Persist ───────────────────────────────────────────────────────
    save_model(best_model, vectorizer)

    return {
        "processed_df": processed_df,
        "vectorizer": vectorizer,
        "models": models,
        "results": results,
        "best_model_name": best_name,
        "best_model": best_model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


# ── Prediction helper ───────────────────────────────────────────────────────
def predict_text(text: str, model=None, vectorizer=None):
    """
    Predict whether a single text is Spam or Ham.

    Parameters
    ----------
    text : str
        Raw message text.
    model : estimator, optional
        If None, loads from disk.
    vectorizer : TfidfVectorizer, optional
        If None, loads from disk.

    Returns
    -------
    dict
        {"label": "Spam"|"Ham", "confidence": float, "probabilities": dict}
    """
    from preprocess import clean_text

    if model is None or vectorizer is None:
        model, vectorizer = load_model()

    cleaned = clean_text(text)
    X = vectorizer.transform([cleaned])
    prediction = model.predict(X)[0]

    # Probability estimates (if available)
    proba = {"Ham": 0.0, "Spam": 0.0}
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[0]
        proba = {"Ham": float(probabilities[0]), "Spam": float(probabilities[1])}

    label = "Spam" if prediction == 1 else "Ham"
    confidence = proba[label]

    return {"label": label, "confidence": confidence, "probabilities": proba}


# ── Save / Load utilities ───────────────────────────────────────────────────
def save_model(model, vectorizer):
    """Persist model and vectorizer to disk via pickle."""
    with open(BEST_MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)


def load_model():
    """Load persisted model and vectorizer from disk."""
    if not os.path.exists(BEST_MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError(
            "No saved model found. Please train a model first."
        )
    with open(BEST_MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


def model_exists() -> bool:
    """Check whether a saved model is available on disk."""
    return os.path.exists(BEST_MODEL_PATH) and os.path.exists(VECTORIZER_PATH)
