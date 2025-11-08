#!/usr/bin/env python3
"""
Defines the LSTM + Attention model for Survival Prediction.
Compatible with TensorFlow 2.15+ masking behavior and stable training.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


def attention_block(inputs, attention_dim):
    """
    Attention mechanism for sequence data.
    inputs: (batch_size, time_steps, hidden_units)
    """
    # Step 1: Compute attention scores
    score = layers.Dense(
        attention_dim,
        activation="tanh",
        kernel_initializer="glorot_uniform",
        name="attn_score_dense_1"
    )(inputs)
    score = layers.Dense(
        1,
        kernel_initializer="glorot_uniform",
        name="attn_score_dense_2"
    )(score)

    # Step 2: Softmax across time dimension
    weights = layers.Softmax(axis=1, name="attention_weights")(score)

    # Step 3: Weighted sum of inputs → context vector
    weighted = layers.Multiply(name="weighted_context")([inputs, weights])
    context_vector = layers.Lambda(
        lambda x: tf.reduce_sum(x, axis=1),
        name="context_vector"
    )(weighted)

    return context_vector, weights


def build_lstm_attention_model(seq_len, n_features,
                               lstm_units=128, attn_dim=64, dropout=0.3):
    """
    Build an LSTM + Attention model for survival prediction.
    """
    inputs = layers.Input(shape=(seq_len, n_features), name="input_seq")

    # Mask padded timesteps (assuming padding value = 0)
    masked = layers.Masking(mask_value=0.0, name="masking")(inputs)

    # Stable Bidirectional LSTM Encoder
    lstm_layer = layers.LSTM(
        lstm_units,
        return_sequences=True,
        dropout=dropout,
        kernel_initializer="glorot_uniform",
        recurrent_initializer="orthogonal"
    )
    lstm_out = layers.Bidirectional(lstm_layer, name="bidirectional_lstm")(masked)

    # Explicitly detach mask to prevent Softmax broadcasting bug
    lstm_out_no_mask = layers.Lambda(lambda x: tf.identity(x), name="detach_mask_explicit")(lstm_out)

    # Attention mechanism
    context_vector, attn_weights = attention_block(lstm_out_no_mask, attn_dim)

    # Fully connected layers
    dense1 = layers.Dense(
        64,
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
        kernel_initializer="glorot_uniform"
    )(context_vector)
    dropout1 = layers.Dropout(dropout)(dense1)
    output = layers.Dense(1, activation="sigmoid", name="survival_prob")(dropout1)

    # Model
    model = models.Model(inputs=inputs, outputs=output, name="LSTM_Attention_Survival")

    # Gradient-safe optimizer (set in train.py)
    model.summary()
    return model