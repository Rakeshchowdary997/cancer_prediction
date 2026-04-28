from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import joblib

# =========================
# 1. Load Dataset
# =========================
data = load_breast_cancer()
X = data.data
y = data.target

# =========================
# 2. Train-Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# =========================
# 3. Feature Scaling
# =========================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# 4. Initialize Models
# =========================
lr = LogisticRegression(max_iter=10000)
rf = RandomForestClassifier(random_state=42)
svm = SVC(probability=True)

# =========================
# 5. Train Models
# =========================
lr.fit(X_train, y_train)
rf.fit(X_train, y_train)
svm.fit(X_train, y_train)

# =========================
# 6. Evaluate Models
# =========================
results = {
    "Logistic Regression": lr.score(X_test, y_test),
    "Random Forest": rf.score(X_test, y_test),
    "SVM": svm.score(X_test, y_test)
}

print("Model Accuracies:")
for k, v in results.items():
    print(f"{k}: {v:.4f}")

# =========================
# 7. Save Models (IMPORTANT)
# =========================

# Individual models
joblib.dump(lr, "model_lr.pkl")
joblib.dump(rf, "model_rf.pkl")
joblib.dump(svm, "model_svm.pkl")

# 🔥 MAIN MODEL (used in app.py)
joblib.dump(rf, "model.pkl")

# Other required files
joblib.dump(scaler, "scaler.pkl")
joblib.dump(results, "model_results.pkl")
joblib.dump(data.feature_names, "feature_names.pkl")

print("\n✅ All models and files saved successfully!")