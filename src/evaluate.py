#!/usr/bin/env python3
"""
Evaluate the trained LSTM + Attention model on test.npz.
This version rebuilds the model architecture and loads weights from the .h5 file,
which avoids HDF5 deserialization issues with internal ops like 'NotEqual'.
"""

import argparse
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, roc_curve, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import os

# import your model builder
from models import build_lstm_attention_model

def load_npz(path):
    """Load .npz dataset and return (X, y, lengths, ids)."""
    data = np.load(path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    lengths = data["lengths"] if "lengths" in data.files else None
    ids = data["ids"] if "ids" in data.files else None
    return X, y, lengths, ids

def evaluate(model_path, test_npz, out_prefix=None):
    print(f"🧾 Loading test data from: {test_npz}")
    X_test, y_test, _, _ = load_npz(test_npz)
    print(f"✅ Test data shape: {X_test.shape}, Labels: {y_test.shape}")

    # Rebuild architecture using the same hyperparameters you used for training.
    seq_len = X_test.shape[1]
    n_features = X_test.shape[2]
    print(f"🔧 Rebuilding model architecture: seq_len={seq_len}, n_features={n_features}")

    model = build_lstm_attention_model(seq_len, n_features, lstm_units=128, attn_dim=64, dropout=0.3)

    # Load weights from the HDF5 file (this typically works even if model.save() wrote full model)
    print(f"📦 Loading weights from: {model_path}")
    try:
        model.load_weights(model_path)
        print("✅ Weights loaded successfully.")
    except Exception as e:
        print("❌ Failed to load weights from the HDF5 file using load_weights().")
        print("Error:", e)
        print("Attempting to fallback to tf.keras.models.load_model (may fail on 'NotEqual' issue)...")
        model = tf.keras.models.load_model(model_path)  # fallback (may raise the same error)

    # Predict
    preds = model.predict(X_test, batch_size=32).ravel()
    pred_labels = (preds >= 0.5).astype(int)

    # Metrics
    auc = roc_auc_score(y_test, preds) if len(np.unique(y_test)) > 1 else float("nan")
    acc = accuracy_score(y_test, pred_labels)
    prec = precision_score(y_test, pred_labels, zero_division=0)
    rec = recall_score(y_test, pred_labels, zero_division=0)
    f1 = f1_score(y_test, pred_labels, zero_division=0)

    print("\n📊 Evaluation Metrics")
    print("---------------------")
    print(f"AUC:        {auc:.4f}")
    print(f"Accuracy:   {acc:.4f}")
    print(f"Precision:  {prec:.4f}")
    print(f"Recall:     {rec:.4f}")
    print(f"F1 Score:   {f1:.4f}")

    cm = confusion_matrix(y_test, pred_labels)
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(y_test, pred_labels, digits=4))

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, preds)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()

    if out_prefix:
        os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
        roc_path = out_prefix + "_roc.png"
        plt.savefig(roc_path, dpi=150)
        print(f"📈 Saved ROC curve at: {roc_path}")
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LSTM + Attention survival model on test set.")
    parser.add_argument("--model", type=str, default="saved_models/survival_rnn.h5", help="Path to saved weights or model file")
    parser.add_argument("--test_npz", type=str, default="data/processed/test.npz", help="Path to test dataset .npz")
    parser.add_argument("--out", type=str, default=None, help="Output prefix for ROC curve (e.g., 'results/eval')")
    args = parser.parse_args()

    evaluate(args.model, args.test_npz, args.out)