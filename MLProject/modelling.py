import os
import mlflow
import mlflow.sklearn
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier

# ===============================
# 1. Load Data
# ===============================
df_path = "data/Telco-Customer-Churn_cleaned.csv"

if not os.path.exists(df_path):
    raise FileNotFoundError(f"❌ File tidak ditemukan di lokasi: {df_path}")

df = pd.read_csv(df_path)
print(f"✅ Dataset berhasil dimuat. Jumlah baris: {len(df)}, kolom: {len(df.columns)}")

# Hapus kolom ID
if "customerID" in df.columns:
    df = df.drop(columns=["customerID"])
    print("🧹 Kolom 'customerID' dihapus.")

# One-hot encoding
df = pd.get_dummies(df, drop_first=True)
print("🔢 Data kategorikal di-encode.")

# Fitur & target
if "Churn_Yes" not in df.columns:
    raise KeyError("❌ Kolom target 'Churn_Yes' tidak ditemukan.")

X = df.drop("Churn_Yes", axis=1)
y = df["Churn_Yes"]

# ===============================
# 2. Train–Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"📊 Train: {len(X_train)}, Test: {len(X_test)}")

# ===============================
# 3. Training (MLflow Project SAFE)
# ===============================

print("📂 Tracking URI aktif:", mlflow.get_tracking_uri())
print("👤 Tracking user:", os.getenv("MLFLOW_TRACKING_USERNAME"))

# ⛔ JANGAN set_experiment
# ⛔ JANGAN start_run
# mlflow run SUDAH MENANGANI RUN-NYA

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)
print("🤖 Model RandomForest dilatih.")

# Evaluasi
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"🎯 Akurasi: {acc:.4f}")

# ===============================
# 4. MLflow Logging
# ===============================

mlflow.log_param("model_type", "RandomForest")
mlflow.log_param("n_estimators", 100)
mlflow.log_metric("accuracy", acc)

# Simpan model
model_path = "model.pkl"
joblib.dump(model, model_path)
mlflow.log_artifact(model_path)

# Classification report
report_path = "classification_report.txt"
with open(report_path, "w") as f:
    f.write(classification_report(y_test, y_pred))

mlflow.log_artifact(report_path)

print("📦 Model, metrik, dan artefak berhasil dicatat ke MLflow.")
print("📋 Classification Report:")
print(classification_report(y_test, y_pred))
