import pandas as pd
import json
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------------
# Paths
# -------------------------------
DATA_PATH = "dataset/winequality-red.csv"
MODEL_PATH = "outputs/model/model.pkl"
RESULT_PATH = "outputs/results/metrics.json"

os.makedirs("outputs/model", exist_ok=True)
os.makedirs("outputs/results", exist_ok=True)

# -------------------------------
# Load Dataset
# -------------------------------
df = pd.read_csv(DATA_PATH, sep=";")

X = df.drop("quality", axis=1)
y = df["quality"]

# -------------------------------
# Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Preprocessing
# -------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------
# Model Training
# -------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -------------------------------
# Evaluation
# -------------------------------
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# -------------------------------
# Save Outputs
# -------------------------------
joblib.dump(model, MODEL_PATH)

metrics = {
    "MSE": mse,
    "R2_score": r2
}

with open(RESULT_PATH, "w") as f:
    json.dump(metrics, f, indent=4)

# -------------------------------
# Print Metrics (Mandatory)
# -------------------------------
print("Mean Squared Error:", mse)
print("R2 Score:", r2)
