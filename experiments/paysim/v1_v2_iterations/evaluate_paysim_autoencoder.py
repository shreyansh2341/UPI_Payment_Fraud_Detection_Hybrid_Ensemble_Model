import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from sklearn.metrics import confusion_matrix, classification_report

DATA_PATH = "Fraud_Detection_Model_Paysim_CC/data/paysim_test.csv"

AE_MODEL = "Fraud_Detection_Model_Paysim_CC/artifacts/paysim_ae_model.keras"
AE_PREPROC = "Fraud_Detection_Model_Paysim_CC/artifacts/paysim_ae_preproc.joblib"
AE_THRESH = "Fraud_Detection_Model_Paysim_CC/artifacts/paysim_ae_thresh.npy"
print("🔍 Evaluating PaySim Autoencoder")

df = pd.read_csv(DATA_PATH)
y = df["isFraud"].values
X_raw = df.drop(columns=["isFraud"])

preproc = joblib.load(AE_PREPROC)
X = preproc.transform(X_raw)

ae = tf.keras.models.load_model(AE_MODEL)
threshold = np.load(AE_THRESH)[0]

recon = ae.predict(X, verbose=0)
errors = np.mean((X - recon) ** 2, axis=1)

y_pred = (errors >= threshold).astype(int)

print("\nAUTOENCODER CONFUSION MATRIX")
print(confusion_matrix(y, y_pred))
print(classification_report(y, y_pred))
