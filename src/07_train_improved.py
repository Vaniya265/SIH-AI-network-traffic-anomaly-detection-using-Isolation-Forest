import pandas as pd
import joblib

from sklearn.ensemble import IsolationForest


# ==========================================
# LOAD NORMAL TRAINING DATA
# ==========================================

X = pd.read_csv(
    "data/normal_processed.csv"
)

print("Training data:", X.shape)


# ==========================================
# IMPROVED ISOLATION FOREST
# ==========================================

model = IsolationForest(
    n_estimators=500,
    max_samples="auto",
    contamination=0.05,
    max_features=0.8,
    bootstrap=False,
    random_state=42,
    n_jobs=-1
)


# ==========================================
# TRAIN
# ==========================================

print("\nTraining improved Isolation Forest...")

model.fit(X)


# ==========================================
# SAVE
# ==========================================

joblib.dump(
    model,
    "models/isolation_forest_improved.pkl"
)


print("\nImproved model trained successfully!")

print(
    "Saved to: models/isolation_forest_improved.pkl"
)