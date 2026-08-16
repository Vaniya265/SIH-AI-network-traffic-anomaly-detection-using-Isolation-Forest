import joblib
import pandas as pd

# Model aur helper files load karo
model = joblib.load("models/isolation_forest_improved.pkl")   # ya isolation_forest.pkl agar wo bolein
scaler = joblib.load("models/scaler.pkl")
feature_columns = joblib.load("models/features.pkl")

# Apni teeno demo files load karo
normal = pd.read_csv("demo_normal.csv")
known = pd.read_csv("demo_known_attack.csv")
hidden = pd.read_csv("demo_hidden_attack.csv")

def prepare_and_predict(data, label_name):
    data = data.drop(columns=["label", "difficulty"], errors="ignore")
    data_encoded = pd.get_dummies(data)
    data_encoded = data_encoded.reindex(columns=feature_columns, fill_value=0)
    scaled = scaler.transform(data_encoded)
    predictions = model.predict(scaled)   # -1 = anomaly/flagged, 1 = normal
    flagged = sum(predictions == -1)
    total = len(predictions)
    print(f"{label_name}: {flagged}/{total} flagged as anomaly")
    return predictions

print("--- Testing ---")
prepare_and_predict(normal, "Normal traffic")
prepare_and_predict(known, "Known attack")
prepare_and_predict(hidden, "Hidden attack (neptune)")