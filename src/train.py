#!/usr/bin/env python3
"""
Train LSTM + Attention model for survival prediction using MIMIC-III time-series data.

Example:
python src/train.py \
    --data_npz data/processed/train.npz \
    --val_npz data/processed/val.npz \
    --model_out saved_models/survival_rnn.h5 \
    --epochs 30 \
    --batch_size 32 \
    --lr 1e-4
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from datetime import datetime

from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
)
from models import build_lstm_attention_model


# -----------------------------
# Utility functions
# -----------------------------

def load_npz(path):
    """Load .npz dataset and return (X, y, lengths, ids)."""
    data = np.load(path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    lengths = data["lengths"] if "lengths" in data.files else None
    ids = data["ids"] if "ids" in data.files else None
    return X, y, lengths, ids


def to_tf_dataset(X, y, batch_size=32, shuffle=True):
    """Convert numpy arrays into a tf.data.Dataset."""
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(X))
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# -----------------------------
# Training logic
# -----------------------------

def train(train_npz, val_npz, model_out, epochs=30, batch_size=32, lr=1e-4):
    # --- Load data ---
    X_train, y_train, _, _ = load_npz(train_npz)
    X_val, y_val, _, _ = load_npz(val_npz)
    seq_len, n_features = X_train.shape[1], X_train.shape[2]

    print(f"📦 Loaded training data: {X_train.shape}, validation data: {X_val.shape}")

    # --- Build model ---
    model = build_lstm_attention_model(seq_len, n_features,
                                       lstm_units=128, attn_dim=64, dropout=0.3)

    # --- Compile model ---
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr,
        clipnorm=1.0  # prevents exploding gradients
    )
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )

    # --- Prepare directories ---
    os.makedirs(os.path.dirname(model_out) or ".", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/train_{timestamp}"

    # --- Callbacks ---
    ckpt = ModelCheckpoint(
        filepath=model_out,
        monitor="val_auc",
        mode="max",
        save_best_only=True,
        verbose=1
    )

    es = EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=6,
        verbose=1,
        restore_best_weights=True
    )

    rlr = ReduceLROnPlateau(
        monitor="val_auc",
        mode="max",
        factor=0.5,
        patience=3,
        verbose=1
    )

    tensorboard = TensorBoard(log_dir=log_dir)

    # --- Convert numpy arrays to datasets ---
    train_ds = to_tf_dataset(X_train, y_train, batch_size=batch_size, shuffle=True)
    val_ds = to_tf_dataset(X_val, y_val, batch_size=batch_size, shuffle=False)

    print("🚀 Starting model training...\n")

    # --- Train ---
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[ckpt, es, rlr, tensorboard],
        verbose=1
    )

    print("\n✅ Training complete.")
    print(f"💾 Best model saved at: {model_out}")
    print(f"🧾 TensorBoard logs: {log_dir}")

    return model, history


# -----------------------------
# Main entry point
# -----------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM + Attention survival model.")
    parser.add_argument("--data_npz", type=str, default="data/processed/train.npz", help="Path to training .npz file")
    parser.add_argument("--val_npz", type=str, default="data/processed/val.npz", help="Path to validation .npz file")
    parser.add_argument("--model_out", type=str, default="saved_models/survival_rnn.h5", help="Path to save best model")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()

    train(args.data_npz, args.val_npz, args.model_out,
          epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)