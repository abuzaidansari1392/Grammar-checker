"""
Error Classifier Module
TF-IDF + Logistic Regression classifier for grammar error types.
Includes training, evaluation, and inference.
"""

import os
import joblib
import numpy as np
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)
from sklearn.pipeline import Pipeline

MODEL_PATH = Path(__file__).parent.parent / "models" / "error_classifier.joblib"
LABELS_PATH = Path(__file__).parent.parent / "models" / "label_mapping.joblib"


# ------------------------------------------------------------------ #
#  Feature engineering
# ------------------------------------------------------------------ #

def _extract_features(sentences: list[str]) -> list[str]:
    """
    Augment raw sentences with hand-crafted features as extra tokens.
    This gives the TF-IDF vectorizer richer signal.
    """
    import re

    augmented = []
    for sent in sentences:
        tokens = sent.lower().split()
        feats = [sent.lower()]  # raw text

        # Presence of third-person pronoun + bare verb
        if any(p in tokens for p in ("he", "she", "it")):
            feats.append("THIRD_PERSON_SUBJECT")

        # Apostrophe contractions → punctuation hint
        if re.search(r"\b\w+'t\b", sent, re.I) or re.search(r"\b\w+'s\b", sent, re.I):
            feats.append("HAS_CONTRACTION")

        # a/an presence → article hint
        if "a" in tokens or "an" in tokens:
            feats.append("HAS_ARTICLE")

        # Common prepositions
        preps = {"in", "on", "at", "by", "for", "with", "about", "of", "from", "to"}
        if preps.intersection(tokens):
            feats.append("HAS_PREPOSITION")

        # Pronoun case markers
        obj_pronouns = {"him", "her", "me", "us", "them"}
        if obj_pronouns.intersection(tokens):
            feats.append("HAS_OBJECT_PRONOUN")

        # Past/future tense markers
        if any(t in tokens for t in ("yesterday", "last", "ago", "went")):
            feats.append("PAST_TIME_MARKER")
        if any(t in tokens for t in ("will", "shall", "tomorrow")):
            feats.append("FUTURE_MARKER")

        augmented.append(" ".join(feats))

    return augmented


# ------------------------------------------------------------------ #
#  Model definition
# ------------------------------------------------------------------ #

def build_pipeline() -> Pipeline:
    """Return the sklearn Pipeline (TF-IDF → Logistic Regression)."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range  = (1, 3),
            min_df       = 1,
            max_features = 8000,
            sublinear_tf = True,
            analyzer     = "word",
        )),
        ("clf", LogisticRegression(
            max_iter     = 1000,
            C            = 5.0,
            solver       = "lbfgs",
            class_weight = "balanced",
            random_state = 42,
        )),
    ])


# ------------------------------------------------------------------ #
#  Training
# ------------------------------------------------------------------ #

def train_classifier(verbose: bool = True) -> dict:
    """
    Train the error classifier on the built-in dataset.
    Returns an evaluation dict with metrics.
    """
    from data.grammar_dataset import get_training_data, get_label_mapping

    sentences, labels = get_training_data()
    label_to_idx, idx_to_label = get_label_mapping()

    # Augment features
    X = _extract_features(sentences)
    y = [label_to_idx[l] for l in labels]

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Build and train
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    label_names = [idx_to_label[i] for i in sorted(idx_to_label)]

    metrics = {
        "accuracy"  : accuracy_score(y_test, y_pred),
        "precision" : precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall"    : recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1"        : f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "conf_matrix"       : confusion_matrix(y_test, y_pred),
        "class_report"      : classification_report(
                                  y_test, y_pred,
                                  target_names=label_names,
                                  zero_division=0,
                              ),
        "label_names"       : label_names,
        "label_to_idx"      : label_to_idx,
        "idx_to_label"      : idx_to_label,
        "n_train"           : len(X_train),
        "n_test"            : len(X_test),
    }

    if verbose:
        print(f"\n=== Classifier Training Results ===")
        print(f"Training samples : {len(X_train)}")
        print(f"Test samples     : {len(X_test)}")
        print(f"Accuracy         : {metrics['accuracy']:.4f}")
        print(f"Precision        : {metrics['precision']:.4f}")
        print(f"Recall           : {metrics['recall']:.4f}")
        print(f"F1 Score         : {metrics['f1']:.4f}")
        print(f"\nClassification Report:\n{metrics['class_report']}")

    # Persist model
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    joblib.dump({"label_to_idx": label_to_idx, "idx_to_label": idx_to_label}, LABELS_PATH)

    metrics["pipeline"] = pipeline
    return metrics


# ------------------------------------------------------------------ #
#  Inference
# ------------------------------------------------------------------ #

_pipeline  = None
_label_map = None


def _load_model():
    global _pipeline, _label_map
    if _pipeline is None:
        if MODEL_PATH.exists() and LABELS_PATH.exists():
            _pipeline  = joblib.load(MODEL_PATH)
            _label_map = joblib.load(LABELS_PATH)
        else:
            # Train on the fly
            result    = train_classifier(verbose=False)
            _pipeline  = result["pipeline"]
            _label_map = {
                "label_to_idx": result["label_to_idx"],
                "idx_to_label": result["idx_to_label"],
            }


def classify_error(sentence: str) -> dict:
    """
    Classify a single sentence into an error category.
    Returns: {label, confidence, probabilities}
    """
    _load_model()
    feat = _extract_features([sentence])
    proba = _pipeline.predict_proba(feat)[0]
    pred_idx = int(np.argmax(proba))
    idx_to_label = _label_map["idx_to_label"]

    label_names = [idx_to_label[i] for i in sorted(idx_to_label)]
    prob_dict   = {idx_to_label[i]: float(proba[i]) for i in range(len(proba))}

    return {
        "label"        : idx_to_label[pred_idx],
        "confidence"   : float(proba[pred_idx]),
        "probabilities": prob_dict,
        "label_names"  : label_names,
    }


def classify_batch(sentences: list[str]) -> list[dict]:
    """Classify a list of sentences."""
    return [classify_error(s) for s in sentences]


def get_model_metrics() -> dict:
    """(Re-)train and return evaluation metrics."""
    return train_classifier(verbose=False)
