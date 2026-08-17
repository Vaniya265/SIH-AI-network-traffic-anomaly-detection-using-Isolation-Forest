import pandas as pd
import joblib

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ==========================================
# 1. LOAD TEST DATA
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


df = pd.read_csv(
    "data/KDDTest+.txt",
    names=columns
)


# ==========================================
# 2. TRUE LABELS
# ==========================================

y_true = (
    df["label"] != "normal"
).astype(int)


# ==========================================
# 3. PREPROCESS
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


feature_names = joblib.load(
    "models/features.pkl"
)

X = X.reindex(
    columns=feature_names,
    fill_value=0
)


scaler = joblib.load(
    "models/scaler.pkl"
)

X_scaled = scaler.transform(X)


# ==========================================
# 4. LOAD MODEL
# ==========================================

model = joblib.load(
    "models/isolation_forest.pkl"
)


# ==========================================
# 5. GET ANOMALY SCORES
# ==========================================

scores = -model.decision_function(X_scaled)


# Higher score = more anomalous


# ==========================================
# 6. TRY MULTIPLE THRESHOLDS
# ==========================================

thresholds = [
    0.00,
    0.02,
    0.04,
    0.06,
    0.08,
    0.10,
    0.12,
    0.14,
    0.16,
    0.18,
    0.20
]


print("\n==============================================")
print("THRESHOLD TUNING")
print("==============================================")

print(
    f"{'Threshold':<12}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'F1':<12}"
)


best_f1 = 0
best_threshold = 0


for threshold in thresholds:

    y_pred = (
        scores >= threshold
    ).astype(int)


    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )


    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )


    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )


    print(
        f"{threshold:<12.2f}"
        f"{precision:<12.4f}"
        f"{recall:<12.4f}"
        f"{f1:<12.4f}"
    )


    if f1 > best_f1:

        best_f1 = f1

        best_threshold = threshold


print("\n==============================================")
print("BEST THRESHOLD")
print("==============================================")

print("Threshold:", best_threshold)
print("F1 Score:", best_f1)


# ==========================================
# 7. CONFUSION MATRIX
# ==========================================

best_predictions = (
    scores >= best_threshold
).astype(int)


cm = confusion_matrix(
    y_true,
    best_predictions
)


print("\nConfusion Matrix:")
print(cm)