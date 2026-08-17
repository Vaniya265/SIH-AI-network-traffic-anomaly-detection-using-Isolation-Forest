import pandas as pd
import joblib

from sklearn.ensemble import IsolationForest


# ==========================================
# 1. LOAD PROCESSED NORMAL DATA
# ==========================================

X = pd.read_csv(
    "data/normal_processed.csv"
)

print("Training data shape:", X.shape)


# ==========================================
# 2. CREATE ISOLATION FOREST
# ==========================================

model = IsolationForest(
    n_estimators=200,
    contamination=0.05,
    random_state=42,
    n_jobs=-1
)


# ==========================================
# 3. TRAIN MODEL
# ==========================================

print("\nTraining Isolation Forest...")

model.fit(X)


# ==========================================
# 4. SAVE MODEL
# ==========================================

joblib.dump(
    model,
    "models/isolation_forest.pkl"
)


print("\nTraining completed!")

print("Model saved to: models/isolation_forest.pkl")