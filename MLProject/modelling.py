import mlflow
import joblib
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import os

# ===============================
# 1. Load Data
# ===============================
df_path = "data/Telco-Customer-Churn_cleaned.csv"

if not os.path.exists(df_path):
    raise FileNotFoundError(f"❌ File tidak ditemukan di lokasi: {df_path}")

df = pd.read_csv(df_path)
print(f"✅ Dataset berhasil dimuat. Jumlah baris: {len(df)}, kolom: {len(df.columns)}")

# Hapus kolom ID unik
if 'customerID' in df.columns:
    df = df.drop(columns=['customerID'])
    print("🧹 Kolom 'customerID' dihapus dari dataset.")

# One-hot encoding
df = pd.get_dummies(df, drop_first=True)
print("🔢 Data kategorikal telah dikonversi menjadi numerik (one-hot encoding).")

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
print(f"📊 Data dibagi menjadi {len(X_train)} train dan {len(X_test)} test.\n")

# ===============================
# 3. MLflow Setup (DAGSHUB)
# ===============================

# --- GANTI BAGIAN INI SESUAI AKUN KAMU ---
username = "mtaufiqs"
repo_name = "Telco-Customer-Churn_ML"        
token = "e112da09dfbf78c022bd120faed1da98ba21c44e"
# ------------------------------------------

# Tracking URI DagsHub
tracking_uri = f"https://dagshub.com/{username}/{repo_name}.mlflow"
mlflow.set_tracking_uri(tracking_uri)

# Set kredensial (harus sebelum start_run)
os.environ["MLFLOW_TRACKING_USERNAME"] = username
os.environ["MLFLOW_TRACKING_PASSWORD"] = token

if os.getenv("MLFLOW_TRACKING_URI"):
    mlflow.set_experiment("Telco-Customer-Churn-DagsHub")
else:
    mlflow.set_experiment("Local-Experiment")

print("📂 Tracking URI aktif:", mlflow.get_tracking_uri())

# ===============================
# 4. Train + MLflow logging
# ===============================
with mlflow.start_run(run_name="Baseline-RandomForest"):

    # Model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("🤖 Model RandomForest berhasil dilatih.")

    # Evaluasi
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"🎯 Akurasi model: {acc:.4f}\n")

    # MLflow Manual Logging
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", acc)

    # Simpan model
    joblib.dump(model, "model.pkl")
    mlflow.log_artifact("model.pkl")

    # Log artefak tambahan
    report_path = "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(classification_report(y_test, y_pred))
    mlflow.log_artifact(report_path)

    print("📦 Model, metrik, dan artefak telah dicatat ke MLflow.\n")

    print("📋 Classification Report:")
    print(classification_report(y_test, y_pred))

