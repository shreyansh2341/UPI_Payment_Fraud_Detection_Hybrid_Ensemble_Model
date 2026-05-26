import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (confusion_matrix, classification_report, roc_auc_score,
                             precision_recall_curve, auc, precision_score, recall_score, f1_score)

LAPTOP_TEST = True
SAMPLE_N = 10000
DATASET = "paysim"   
DATA_DIR = Path("data")
PAYSIM_FILE = DATA_DIR / "cleaned_paysim.csv"
CREDIT_FILE = DATA_DIR / "cleaned_creditcard.csv"
OUTPUT_DIR = Path("artifacts")
OUTPUT_DIR.mkdir(exist_ok=True)
RANDOM_STATE = 42
# AE params
if LAPTOP_TEST:
    AE_EPOCHS = 8
    AE_BATCH = 256
    AE_LATENT = 8
else:
    AE_EPOCHS = 50
    AE_BATCH = 4096
    AE_LATENT = 32

def load_dataset(name="paysim", laptop_mode=True, sample_n=10000):
    if name == "paysim":
        path = PAYSIM_FILE
    elif name == "creditcard":
        path = CREDIT_FILE
    else:
        raise ValueError("Dataset must be 'paysim' or 'creditcard'")
    df = pd.read_csv(path)
    if laptop_mode:
        df = df.sample(min(sample_n, len(df)), random_state=RANDOM_STATE).reset_index(drop=True)
    print(f"Loaded {name} shape:", df.shape)
    return df

def split_df(df):
    X = df.drop(columns=['isFraud'])
    y = df['isFraud'].astype(int)
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

def build_preprocessor():
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())   # important for AE
    ])

def build_autoencoder(input_dim, latent_dim=16):
    inp = layers.Input(shape=(input_dim,))
    x = layers.Dense(latent_dim*4, activation='relu')(inp)
    x = layers.Dense(latent_dim*2, activation='relu')(x)
    encoded = layers.Dense(latent_dim, activation='relu')(x)
    x = layers.Dense(latent_dim*2, activation='relu')(encoded)
    x = layers.Dense(latent_dim*4, activation='relu')(x)
    out = layers.Dense(input_dim, activation='linear')(x)
    ae = models.Model(inp, out)
    ae.compile(optimizer='adam', loss='mse')
    return ae

def compute_ae_features(ae, preproc, X_df, thresh=None):
    Xp = preproc.transform(X_df)
    recon = ae.predict(Xp, verbose=0)
    errs = np.mean((Xp - recon)**2, axis=1)
    if thresh is None:
        return errs
    flags = (errs > thresh).astype(int)
    return errs, flags

def evaluate_model(model, X_te_t, y_te, name="model"):
    proba = model.predict_proba(X_te_t)[:,1]
    pred = (proba >= 0.5).astype(int)
    print(f"\n=== {name} ===")
    print("Confusion matrix:\n", confusion_matrix(y_te, pred))
    print(classification_report(y_te, pred, digits=4))
    roc = roc_auc_score(y_te, proba)
    prec, rec, _ = precision_recall_curve(y_te, proba)
    pr_auc = auc(rec, prec)
    print(f"ROC-AUC={roc:.4f}  PR-AUC={pr_auc:.4f}")
    return {'roc_auc': roc, 'pr_auc': pr_auc,
            'precision': precision_score(y_te, pred),
            'recall': recall_score(y_te, pred),
            'f1': f1_score(y_te, pred)}

def main():
    df = load_dataset(DATASET, LAPTOP_TEST, SAMPLE_N)
    X_tr, X_te, y_tr, y_te = split_df(df)

    preproc = build_preprocessor()
    X_tr_p = preproc.fit_transform(X_tr)
    X_te_p = preproc.transform(X_te)

    X_tr_nf = X_tr_p[y_tr.values == 0]
    ae = build_autoencoder(X_tr_p.shape[1], AE_LATENT)
    es = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    ae.fit(X_tr_nf, X_tr_nf, epochs=AE_EPOCHS, batch_size=AE_BATCH,
           validation_split=0.1, callbacks=[es], verbose=1)

    errs_nf = compute_ae_features(ae, preproc, X_tr[y_tr==0])
    thresh = np.percentile(errs_nf, 95)
    print("AE threshold (95th percentile non-fraud errors):", thresh)

    ae_err_tr, ae_flag_tr = compute_ae_features(ae, preproc, X_tr, thresh)
    ae_err_te, ae_flag_te = compute_ae_features(ae, preproc, X_te, thresh)

    X_tr_aug = X_tr.copy()
    X_tr_aug['ae_error'] = ae_err_tr
    X_tr_aug['ae_anomaly'] = ae_flag_tr
    X_te_aug = X_te.copy()
    X_te_aug['ae_error'] = ae_err_te
    X_te_aug['ae_anomaly'] = ae_flag_te

    # 6) Preprocess again for RF/XGB (robust scaler works well)
    sup_preproc = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])
    X_tr_sup = sup_preproc.fit_transform(X_tr_aug)
    X_te_sup = sup_preproc.transform(X_te_aug)

    rf = RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_tr_sup, y_tr)

    pos = y_tr.sum(); neg = len(y_tr) - pos
    spw = neg / (pos+1e-9)
    xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                        scale_pos_weight=spw, use_label_encoder=False,
                        eval_metric='logloss', random_state=RANDOM_STATE, n_jobs=-1)
    xgb.fit(X_tr_sup, y_tr)

    m_rf = evaluate_model(rf, X_te_sup, y_te, "RF+AE")
    m_xgb = evaluate_model(xgb, X_te_sup, y_te, "XGB+AE")

    print("\nComparison:")
    print(pd.DataFrame([m_rf, m_xgb], index=['RF+AE','XGB+AE']))

    joblib.dump(ae, OUTPUT_DIR / f"{DATASET}_ae_model.keras")  # keras model
    joblib.dump(preproc, OUTPUT_DIR / f"{DATASET}_ae_preproc.joblib")
    joblib.dump(sup_preproc, OUTPUT_DIR / f"{DATASET}_sup_preproc.joblib")
    joblib.dump(rf, OUTPUT_DIR / f"{DATASET}_rf_ae.joblib")
    joblib.dump(xgb, OUTPUT_DIR / f"{DATASET}_xgb_ae.joblib")
    np.save(OUTPUT_DIR / f"{DATASET}_ae_thresh.npy", np.array([thresh]))
    print("Saved AE, preprocessors, RF, and XGB models with AE features.")

if __name__ == "__main__":
    main()

