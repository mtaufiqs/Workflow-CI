import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import os

# ===== 1. Load data =====
df_path = df_path = "data/Telco-Customer-Churn_cleaned.csv"

if not os.path.exists(df_path):
    raise FileNotFoundError(f"❌ File tidak ditemukan di lokasi: {df_path}")

df = pd.read_csv(df_path)
print(f"✅ Dataset berhasil dimuat. Jumlah baris: {len(df)}, kolom: {len(df.columns)}")

# Hapus kolom ID unik yang tidak berguna
if 'customerID' in df.columns:
    df = df.drop(columns=['customerID'])
    print("🧹 Kolom 'customerID' dihapus dari dataset.")

# Ubah variabel kategorikal menjadi numerik (one-hot encoding)
df = pd.get_dummies(df, drop_first=True)
print("🔢 Data kategorikal telah dikonversi menjadi numerik (one-hot encoding).")

# Pisahkan fitur dan target
if "Churn_Yes" not in df.columns:
    raise KeyError("❌ Kolom target 'Churn_Yes' tidak ditemukan. Pastikan preprocessing benar.")
X = df.drop("Churn_Yes", axis=1)
y = df["Churn_Yes"]

# ===== 2. Split data =====
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"📊 Data dibagi menjadi {len(X_train)} data train dan {len(X_test)} data test.\n")

# ===== 3. Setup MLflow =====
# Pastikan mlruns berada di dalam folder Membangun_Model
project_dir = r"C:\Users\User\Downloads\SMSML_Muhammad-Taufiqurrahman\Membangun_Model"
mlruns_path = f"file:///{project_dir}/mlruns"

mlflow.set_tracking_uri(mlruns_path)
mlflow.set_experiment("Telco Customer Churn")

# Tampilkan lokasi aktif untuk verifikasi
print("📂 Tracking URI aktif:", mlflow.get_tracking_uri())

with mlflow.start_run(run_name="Baseline RandomForest"):
    # ===== 4. Model =====
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("🤖 Model RandomForest berhasil dilatih.")

    # ===== 5. Evaluasi =====
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"🎯 Akurasi model: {acc:.4f}\n")

    # ===== 6. Log MLflow =====
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "model")

    print("📦 Model dan metrik telah dicatat ke MLflow.\n")

    # ===== 7. Tampilkan hasil akhir =====
    print("📋 Classification Report:")

    print(classification_report(y_test, y_pred))
