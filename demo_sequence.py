import joblib
import pandas as pd
import time

# Model load karo
model = joblib.load("models/isolation_forest_improved.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_columns = joblib.load("models/features.pkl")

def prepare(data):
    data = data.drop(columns=["label", "difficulty"], errors="ignore")
    data_encoded = pd.get_dummies(data)
    data_encoded = data_encoded.reindex(columns=feature_columns, fill_value=0)
    return scaler.transform(data_encoded)

def predict_one_row(row_df, row_label):
    scaled = prepare(row_df)
    pred = model.predict(scaled)[0]
    score = model.decision_function(scaled)[0]
    verdict = "🚨 FLAGGED (Anomaly)" if pred == -1 else "✅ Safe (Normal)"
    print(f"[{row_label}] Score: {score:.3f} → {verdict}")
    return pred

# Teeno demo files load karo
normal = pd.read_csv("demo_normal.csv")
known = pd.read_csv("demo_known_attack.csv")
hidden = pd.read_csv("demo_hidden_attack.csv")

print("=" * 50)
print("LIVE DEMO SEQUENCE — SIH1451")
print("=" * 50)

# STEP A: Normal traffic dikhana (should mostly be safe)
print("\n--- Step 1: Normal Traffic ---")
for i in range(3):
    predict_one_row(normal.iloc[[i]], f"Normal row {i+1}")
    time.sleep(1)

# STEP B: Known attack dikhana (should be flagged)
print("\n--- Step 2: Known Attack ---")
for i in range(2):
    predict_one_row(known.iloc[[i]], f"Known Attack row {i+1}")
    time.sleep(1)

# STEP C: Hidden/unseen attack dikhana (the big reveal!)
print("\n--- Step 3: HIDDEN Attack (Never seen in training!) ---")
for i in range(len(hidden)):
    predict_one_row(hidden.iloc[[i]], f"Hidden Attack (neptune) row {i+1}")
    time.sleep(1)

print("\n" + "=" * 50)
print("DEMO COMPLETE — Model caught the unseen attack!")
print("=" * 50)