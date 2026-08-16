import joblib
import pandas as pd

model = joblib.load("models/isolation_forest_improved.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_columns = joblib.load("models/features.pkl")

def prepare(data):
    data = data.drop(columns=["label", "difficulty"], errors="ignore")
    data_encoded = pd.get_dummies(data)
    data_encoded = data_encoded.reindex(columns=feature_columns, fill_value=0)
    return scaler.transform(data_encoded)

normal = pd.read_csv("demo_normal.csv")
known = pd.read_csv("demo_known_attack.csv")
hidden = pd.read_csv("demo_hidden_attack.csv")

def get_scores(data):
    scaled = prepare(data)
    return model.decision_function(scaled)   # negative = zyada suspicious

normal_scores = get_scores(normal)
known_scores = get_scores(known)
hidden_scores = get_scores(hidden)

print("Normal scores:", normal_scores)
print("Known attack scores:", known_scores)
print("Hidden attack scores:", hidden_scores)

# Alag-alag thresholds try karo
for threshold in [0.0, 0.02, 0.05, 0.08, 0.1]:
    normal_flagged = sum(normal_scores < threshold)
    known_flagged = sum(known_scores < threshold)
    hidden_flagged = sum(hidden_scores < threshold)
    print(f"\nThreshold {threshold}:")
    print(f"  Normal false alarms: {normal_flagged}/{len(normal_scores)}")
    print(f"  Known attack caught: {known_flagged}/{len(known_scores)}")
    print(f"  Hidden attack caught: {hidden_flagged}/{len(hidden_scores)}")