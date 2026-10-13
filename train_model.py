"""Training and Evaluation script for Book Genre Classification Using Multinomial Naive Bayes.
College Information Retrieval Project.

This script:
1. Loads dataset.csv (200 curated demonstration books across 8 genres).
2. Performs a stratified 80/20 train/test split to prevent data leakage.
3. Applies CountVectorizer with English stop-word removal.
4. Trains a Multinomial Naive Bayes classifier with Laplace smoothing (alpha=1.0).
5. Evaluates model performance on the held-out test set (Accuracy, Precision, Recall, F1, Confusion Matrix).
6. Saves model artifacts (model.joblib, vectorizer.joblib, and evaluation_metrics.json).
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

def train_and_evaluate():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, "dataset.csv")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    print(f"[*] Loading dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path)
    print(f"[+] Loaded {len(df)} samples across {df['genre'].nunique()} genres.")

    # Combine Title and Description for text classification
    df["full_text"] = df["title"].astype(str) + " " + df["description"].astype(str)

    X = df["full_text"]
    y = df["genre"]

    # Stratified Train-Test Split (80% Train, 20% Test)
    # 20 samples per genre for training (160 total), 5 per genre for test (40 total)
    X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
        X, y, df.index, test_size=0.20, random_state=42, stratify=y
    )

    print(f"[+] Train set size: {len(X_train)} samples ({len(X_train) // 8} per genre)")
    print(f"[+] Test set size:  {len(X_test)} samples ({len(X_test) // 8} per genre)")

    # Vectorization using CountVectorizer (Bag of Words / Term Frequencies)
    print("[*] Vectorizing text with CountVectorizer (English stop-words removed)...")
    vectorizer = CountVectorizer(stop_words="english", lowercase=True, min_df=1)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    vocab_size = len(vectorizer.vocabulary_)
    print(f"[+] Vocabulary size |V|: {vocab_size} unique terms")

    # Train Multinomial Naive Bayes with Laplace Smoothing (alpha = 1.0)
    print("[*] Training Multinomial Naive Bayes (alpha=1.0)...")
    nb_classifier = MultinomialNB(alpha=1.0)
    nb_classifier.fit(X_train_vec, y_train)

    # Evaluate on held-out test set
    print("[*] Evaluating on held-out test set...")
    y_pred = nb_classifier.predict(X_test_vec)
    y_proba = nb_classifier.predict_proba(X_test_vec)

    acc = float(accuracy_score(y_test, y_pred))
    prec_macro = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

    prec_weighted = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
    rec_weighted = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
    f1_weighted = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

    clf_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=nb_classifier.classes_).tolist()

    print("\n" + "=" * 50)
    print("        MODEL EVALUATION RESULTS (TEST SET)")
    print("=" * 50)
    print(f"Accuracy:           {acc * 100:.2f}%")
    print(f"Macro Precision:    {prec_macro * 100:.2f}%")
    print(f"Macro Recall:       {rec_macro * 100:.2f}%")
    print(f"Macro F1-Score:     {f1_macro * 100:.2f}%")
    print(f"Weighted F1-Score:  {f1_weighted * 100:.2f}%")
    print("=" * 50)
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Calculate actual Class Prior Probabilities P(C_k)
    priors = {
        cls: float(np.exp(log_prior))
        for cls, log_prior in zip(nb_classifier.classes_, nb_classifier.class_log_prior_)
    }

    # Store test set samples for live demonstration inspection in Streamlit
    test_df = df.loc[test_indices].copy()
    test_df["predicted_genre"] = y_pred
    test_df["is_correct"] = test_df["genre"] == test_df["predicted_genre"]
    test_samples = test_df[
        ["title", "description", "genre", "predicted_genre", "is_correct"]
    ].to_dict(orient="records")

    metrics_payload = {
        "accuracy": acc,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
        "precision_weighted": prec_weighted,
        "recall_weighted": rec_weighted,
        "f1_weighted": f1_weighted,
        "classification_report": clf_report,
        "confusion_matrix": cm,
        "classes": list(nb_classifier.classes_),
        "class_priors": priors,
        "vocabulary_size": vocab_size,
        "total_train_samples": int(len(X_train)),
        "total_test_samples": int(len(X_test)),
        "total_samples": int(len(df)),
        "alpha_smoothing": 1.0,
        "test_samples": test_samples,
    }

    # Save artifacts
    model_file = os.path.join(script_dir, "model.joblib")
    vectorizer_file = os.path.join(script_dir, "vectorizer.joblib")
    metrics_file = os.path.join(script_dir, "evaluation_metrics.json")

    print("[*] Saving model artifacts...")
    joblib.dump(nb_classifier, model_file)
    joblib.dump(vectorizer, vectorizer_file)
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    print(f"[+] Saved model to {model_file}")
    print(f"[+] Saved vectorizer to {vectorizer_file}")
    print(f"[+] Saved evaluation metrics to {metrics_file}")
    print("[+] Model training and evaluation successfully completed!")

if __name__ == "__main__":
    train_and_evaluate()
