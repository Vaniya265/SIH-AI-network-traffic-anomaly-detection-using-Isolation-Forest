import pandas as pd
import numpy as np
import joblib


# ==========================================
# LOAD NORMAL TRAINING DATA
# ==========================================

X = pd.read_csv(
    "data/normal_processed.csv"
)

print("Training data:", X.shape)


# ==========================================
# LOAD MODEL
# ==========================================

model = joblib.load(
    "models/isolation_forest_improved.pkl"
)


# ==========================================
# CALCULATE NORMAL SCORES
# ==========================================

normal_scores = -model.decision_function(X)


# ==========================================
# CALCULATE SCORE BOUNDARIES
# ==========================================

p5 = np.percentile(
    normal_scores,
    5
)

p50 = np.percentile(
    normal_scores,
    50
)

p95 = np.percentile(
    normal_scores,
    95
)

p99 = np.percentile(
    normal_scores,
    99
)


print("\n========================================")
print("NORMAL SCORE DISTRIBUTION")
print("========================================")

print(f"5th percentile  : {p5:.6f}")
print(f"50th percentile : {p50:.6f}")
print(f"95th percentile : {p95:.6f}")
print(f"99th percentile : {p99:.6f}")


# ==========================================
# SAVE CALIBRATION
# ==========================================

calibration = {
    "p5": float(p5),
    "p50": float(p50),
    "p95": float(p95),
    "p99": float(p99)
}


joblib.dump(
    calibration,
    "models/risk_calibration.pkl"
)


print("\nCalibration saved to:")
print("models/risk_calibration.pkl")