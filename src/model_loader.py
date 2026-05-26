"""
model_loader.py — V3 + V4 Hybrid
Loads all model artifacts for the fraud detection system.

V3: 19-feature ensemble (XGB+RF with AE error as 19th feature)
V4: 20-feature ensemble (XGB+RF with AE error + sequential score)
"""
import joblib
import numpy as np
import tensorflow as tf
from pathlib import Path

# Monkey patch Keras to ignore newer 'quantization_config' args
try:
    from keras.src.layers.core.dense import Dense
    _original_dense_from_config = Dense.from_config
    @classmethod
    def _patched_dense_from_config(cls, config):
        config.pop('quantization_config', None)
        return _original_dense_from_config(config)
    Dense.from_config = _patched_dense_from_config
except ImportError:
    pass

tf.get_logger().setLevel('ERROR')

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"
PAYSIM_V3_DIR = MODELS_DIR / "paysim_v3"
PAYSIM_V4_DIR = MODELS_DIR / "paysim_v4_experiment"
CC_DIR = MODELS_DIR / "creditcard"


def load_paysim_hybrid():
    """Load V3 hybrid models: XGB+RF (19 features) + AE + IForest."""
    d = PAYSIM_V3_DIR
    return {
        "xgb": joblib.load(d / "paysim_v3_xgb.pkl"),
        "rf": joblib.load(d / "paysim_v3_rf.pkl"),
        "ae": tf.keras.models.load_model(d / "paysim_v3_ae.keras", compile=False),
        "iforest": joblib.load(d / "paysim_v3_iforest.pkl"),
        "scaler": joblib.load(d / "paysim_v3_scaler.pkl"),
        "features": joblib.load(d / "paysim_v3_features.pkl"),
        "block_threshold": float(np.load(d / "paysim_v3_threshold.npy")[0]),
        "ae_threshold": float(np.load(d / "paysim_v3_ae_threshold.npy")[0]),
        "weights": np.load(d / "paysim_v3_weights.npy"),
    }


def load_paysim_v4_hybrid():
    """
    Load V4 hybrid models: XGB+RF (20 features) + AE + Sequential + IForest.

    V4 adds a sequential model (BiLSTM or BiGRU with Attention) whose
    fraud probability score becomes the 20th feature for XGBoost/RF.
    Path B is also enhanced with sequential anomaly detection.

    Returns None if V4 models don't exist (experiment not run yet).
    """
    d = PAYSIM_V4_DIR

    if not d.exists():
        print("⚠️  V4 models not found. Run train_paysim_v4_hybrid.py first.")
        return None

    # Import custom Attention layer for model deserialization
    from src.v4_layers import BahdanauAttention

    # Load seq_block_threshold (added by three-tier retuning script)
    seq_block_path = d / "paysim_v4_seq_block_threshold.npy"
    seq_block_threshold = (
        float(np.load(seq_block_path)[0])
        if seq_block_path.exists()
        else 0.5  # Default fallback before retuning
    )

    return {
        # Supervised models (20 features: 18 base + ae_error + seq_score)
        "xgb": joblib.load(d / "paysim_v4_xgb.pkl"),
        "rf": joblib.load(d / "paysim_v4_rf.pkl"),

        # Autoencoder (same as V3 — trained on 18 base features)
        "ae": tf.keras.models.load_model(d / "paysim_v4_ae.keras", compile=False),

        # Sequential model (BiLSTM or BiGRU — the winner)
        "sequential": tf.keras.models.load_model(
            d / "paysim_v4_sequential_winner.keras",
            compile=False,
            custom_objects={"BahdanauAttention": BahdanauAttention},
        ),

        # Isolation Forest
        "iforest": joblib.load(d / "paysim_v4_iforest.pkl"),

        # Scalers
        "base_scaler": joblib.load(d / "paysim_v4_base_scaler.pkl"),  # 18 features
        "scaler": joblib.load(d / "paysim_v4_scaler.pkl"),            # 20 features

        # Feature lists
        "features": joblib.load(d / "paysim_v4_features.pkl"),        # 18 base
        "features_20": joblib.load(d / "paysim_v4_features_20.pkl"),  # 20 full

        # Thresholds (three-tier)
        "block_threshold": float(np.load(d / "paysim_v4_threshold.npy")[0]),
        "ae_threshold": float(np.load(d / "paysim_v4_ae_threshold.npy")[0]),
        "seq_block_threshold": seq_block_threshold,       # Tier 2: novel fraud blocking
        "seq_threshold": float(np.load(d / "paysim_v4_seq_threshold.npy")[0]),  # Tier 3: review
        "weights": np.load(d / "paysim_v4_weights.npy"),

        # Config
        "seq_length": joblib.load(d / "paysim_v4_seq_length.pkl"),
    }


def load_creditcard_models():
    """Load Credit Card ensemble (XGBoost + Random Forest)."""
    return {
        "xgb": joblib.load(CC_DIR / "cc_xgb_model.pkl"),
        "rf": joblib.load(CC_DIR / "cc_rf_model.pkl"),
        "scaler": joblib.load(CC_DIR / "cc_scaler.pkl"),
        "features": joblib.load(CC_DIR / "cc_features.pkl"),
        "weights": np.load(CC_DIR / "cc_ensemble_weights.npy"),
        "threshold": float(np.load(CC_DIR / "cc_ensemble_threshold.npy").item()),
    }


def load_v5_hybrid():
    """
    Load V5 hybrid: V3 models for Path A + V4 models for Path B.

    Returns dict with 'v3' and 'v4' sub-dicts, or None if models missing.
    """
    v3 = load_paysim_hybrid()
    v4 = load_paysim_v4_hybrid()

    if v3 is None or v4 is None:
        print("⚠️  V5 hybrid requires both V3 and V4 models.")
        return None

    return {"v3": v3, "v4": v4}