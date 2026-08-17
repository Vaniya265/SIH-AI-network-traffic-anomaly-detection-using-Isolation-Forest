import pandas as pd
import joblib


# ==========================================
# 1. COLUMN NAMES
# ==========================================

columns = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "label",
    "difficulty"
]


# ==========================================
# 2. LOAD TEST DATA
# ==========================================

df = pd.read_csv(
    "data/KDDTest+.txt",
    names=columns
)

print("Test dataset shape:", df.shape)


# ==========================================
# 3. SAVE TRUE LABELS
# ==========================================

true_labels = df["label"].copy()


# ==========================================
# 4. REMOVE LABEL + DIFFICULTY
# ==========================================

X = df.drop(
    columns=["label", "difficulty"]
)


# ==========================================
# 5. ONE-HOT ENCODE
# ==========================================

categorical_columns = [
    "protocol_type",
    "service",
    "flag"
]

X = pd.get_dummies(
    X,
    columns=categorical_columns
)


# ==========================================
# 6. LOAD TRAINING FEATURE LIST
# ==========================================

feature_names = joblib.load(
    "models/features.pkl"
)


# ==========================================
# 7. ALIGN TEST FEATURES WITH TRAINING
# ==========================================

X = X.reindex(
    columns=feature_names,
    fill_value=0
)


# ==========================================
# 8. LOAD SCALER
# ==========================================

scaler = joblib.load(
    "models/scaler.pkl"
)


# ==========================================
# 9. SCALE TEST DATA
# ==========================================

X_scaled = scaler.transform(X)


# ==========================================
# 10. LOAD ISOLATION FOREST
# ==========================================

model = joblib.load(
    "models/isolation_forest.pkl"
)


# ==========================================
# 11. PREDICT
# ==========================================

predictions = model.predict(X_scaled)


# Isolation Forest:
#  1  = normal
# -1  = anomaly

predicted_labels = [
    "normal" if prediction == 1 else "anomaly"
    for prediction in predictions
]


# ==========================================
# 12. ADD RESULTS
# ==========================================

results = pd.DataFrame({
    "actual": true_labels,
    "predicted": predicted_labels
})


# ==========================================
# 13. DISPLAY RESULTS
# ==========================================

print("\nPrediction distribution:")

print(
    results["predicted"].value_counts()
)


print("\nSample predictions:")

print(
    results.head(20)
)


# ==========================================
# 14. SAVE RESULTS
# ==========================================

results.to_csv(
    "data/detection_results.csv",
    index=False
)

print("\nResults saved to:")
print("data/detection_results.csv")