import joblib
import pandas as pd
import numpy as np

# -----------------------------
# Load trained model & files
# -----------------------------
model = joblib.load("models/isolation_forest_improved.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_columns = joblib.load("models/features.pkl")
calibration = joblib.load("models/risk_calibration.pkl")


# -----------------------------
# Prepare input data
# -----------------------------
def prepare(data):
    data = data.drop(columns=["label", "difficulty"], errors="ignore")
    data_encoded = pd.get_dummies(data)
    data_encoded = data_encoded.reindex(columns=feature_columns, fill_value=0)
    scaled = scaler.transform(data_encoded)
    return pd.DataFrame(scaled, columns=feature_columns, index=data_encoded.index)


# -----------------------------
# Calculate risk score (calibrated)
# -----------------------------
def calculate_risk_score(anomaly_score):
    p95 = calibration["p95"]
    p99 = calibration["p99"]

    # LOW RISK
    if anomaly_score <= p95:
        return 20.0

    # MEDIUM RISK
    elif anomaly_score <= p99:
        ratio = (anomaly_score - p95) / (p99 - p95)
        score = 20 + (ratio * 50)
        return round(float(np.clip(score, 20, 70)), 2)

    # HIGH RISK
    else:
        excess = anomaly_score - p99
        score = 70 + 30 * (1 - np.exp(-5 * excess))
        return round(float(np.clip(score, 70, 100)), 2)


# -----------------------------
# Calculate verdict
# -----------------------------
def get_verdict(row_df, row_label):
    prepared_data = prepare(row_df)

    raw_score = model.decision_function(prepared_data)[0]
    anomaly_score = -raw_score              # SIGN FLIP: negative raw_score = high anomaly
    risk_score = calculate_risk_score(anomaly_score)

    if risk_score >= 70:
        verdict = "Compromised"
    elif risk_score >= 40:
        verdict = "Suspicious"
    else:
        verdict = "Safe"

    print(f"[{row_label}] Raw Score: {raw_score:.3f} | Risk Score: {risk_score}/100 | Verdict: {verdict}")

    return {
        "raw_score": round(raw_score, 3),
        "risk_score": risk_score,
        "verdict": verdict
    }


# -----------------------------
# Load test datasets
# -----------------------------
normal = pd.read_csv("demo_normal.csv")
known = pd.read_csv("demo_known_attack.csv")
hidden = pd.read_csv("demo_hidden_attack.csv")


# -----------------------------
# Run tests
# -----------------------------
print("--- Normal ---")
for i in range(len(normal)):
    get_verdict(normal.iloc[[i]], f"Normal {i + 1}")

print("\n--- Known Attack ---")
for i in range(len(known)):
    get_verdict(known.iloc[[i]], f"Known {i + 1}")

print("\n--- Hidden Attack (neptune) ---")
for i in range(len(hidden)):
    get_verdict(hidden.iloc[[i]], f"Hidden {i + 1}")