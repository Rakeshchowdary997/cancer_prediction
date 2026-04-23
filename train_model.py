from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import joblib

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Models
lr = LogisticRegression(max_iter=10000)
rf = RandomForestClassifier()
svm = SVC(probability=True)

# Train
lr.fit(X_train, y_train)
rf.fit(X_train, y_train)
svm.fit(X_train, y_train)

# Accuracy
results = {
    "Logistic Regression": lr.score(X_test, y_test),
    "Random Forest": rf.score(X_test, y_test),
    "SVM": svm.score(X_test, y_test)
}

# Save everything
joblib.dump(lr, "model_lr.pkl")
joblib.dump(rf, "model_rf.pkl")
joblib.dump(svm, "model_svm.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(results, "model_results.pkl")
joblib.dump(data.feature_names, "feature_names.pkl")

print("All models saved successfully!")