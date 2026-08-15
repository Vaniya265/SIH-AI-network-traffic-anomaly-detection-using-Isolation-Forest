import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


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

print("Test data:", df.shape)


# ==========================================
# 3. TRUE LABELS
# ==========================================

y_true = (
    df["label"] != "normal"
).astype(int)


# ==========================================
# 4. PREPROCESS TEST DATA
# ==========================================

X = df.drop(
    columns=["label", "difficulty"]
)

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
# 5. ALIGN FEATURES
# ==========================================

feature_names = joblib.load(
    "models/features.pkl"
)

X = X.reindex(
    columns=feature_names,
    fill_value=0
)


# ==========================================
# 6. SCALE
# ==========================================

scaler = joblib.load(
    "models/scaler.pkl"
)

X_scaled = scaler.transform(X)


# ==========================================
# 7. LOAD IMPROVED MODEL
# ==========================================

model = joblib.load(
    "models/isolation_forest_improved.pkl"
)


# ==========================================
# 8. PREDICT
# ==========================================

predictions = model.predict(X_scaled)

y_pred = (
    predictions == -1
).astype(int)


# ==========================================
# 9. METRICS
# ==========================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred
)

recall = recall_score(
    y_true,
    y_pred
)

f1 = f1_score(
    y_true,
    y_pred
)


print("\n========================================")
print("IMPROVED MODEL PERFORMANCE")
print("========================================")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")


# ==========================================
# 10. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_true,
    y_pred
)

print("\n========================================")
print("CONFUSION MATRIX")
print("========================================")

print(cm)

print("\nFormat:")
print("[[True Normal, False Attack]]")
print("[[Missed Attack, Detected Attack]]")