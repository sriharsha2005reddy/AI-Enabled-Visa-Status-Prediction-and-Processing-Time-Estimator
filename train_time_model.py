import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("milestone2.csv")

# REMOVE case_status from features
X = df.drop(["processing_time", "case_status"], axis=1)
y = df["processing_time"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

pickle.dump(model, open("model.pkl", "wb"))

print("Time Model Trained Successfully (13 features)")
print("Saved → model.pkl")
