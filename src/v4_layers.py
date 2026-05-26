"""
v4_layers.py — Custom Keras Layers for V4 Pipeline
═══════════════════════════════════════════════════
Contains the custom layers needed for loading V4 sequential models.
Separated from the training script so they can be imported without
triggering the full training pipeline.
"""
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense


class BahdanauAttention(Layer):
    """
    Bahdanau (Additive) Attention Mechanism.

    Input:  (batch_size, timesteps, features) — sequence of hidden states
    Output: (batch_size, features) — context vector (weighted sum of states)
    """

    def __init__(self, units=32, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.W = Dense(units, use_bias=True, name="attention_projection")
        self.V = Dense(1, use_bias=False, name="attention_score")

    def call(self, hidden_states):
        projected = tf.keras.activations.tanh(self.W(hidden_states))
        score = self.V(projected)
        attention_weights = tf.keras.activations.softmax(score, axis=1)
        context_vector = tf.reduce_sum(hidden_states * attention_weights, axis=1)
        return context_vector

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


def focal_loss(alpha=0.75, gamma=2.0):
    """Focal loss for class-imbalanced binary classification."""
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
        modulating_factor = tf.pow(1 - p_t, gamma)
        return alpha_factor * modulating_factor * bce
    return loss
