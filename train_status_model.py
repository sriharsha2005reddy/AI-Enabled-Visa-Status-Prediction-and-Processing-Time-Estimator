import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load preprocessed dataset
df = pd.read_csv("milestone2.csv")

# Target Column
y = df["case_status"]

# Features (remove case_status + processing_time)
X = df.drop(["case_status", "processing_time"], axis=1)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Predictions
pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, pred))
print("\nClassification Report:\n", classification_report(y_test, pred))

# Save Model
joblib.dump(model, "status_model.pkl")

print("\nstatus_model.pkl saved successfully!")
