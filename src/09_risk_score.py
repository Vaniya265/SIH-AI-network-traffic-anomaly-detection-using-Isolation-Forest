import pandas as pd
import joblib
import numpy as np


# ==========================================
# LOAD DATA
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
# PREPROCESS
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
# LOAD MODEL
# ==========================================

model = joblib.load(
    "models/isolation_forest_improved.pkl"
)


# ==========================================
# ANOMALY SCORES
# ==========================================

raw_scores = -model.decision_function(
    X_scaled
)


# ==========================================
# NORMALIZE TO 0–100
# ==========================================

min_score = raw_scores.min()
max_score = raw_scores.max()


risk_scores = (
    (raw_scores - min_score)
    /
    (max_score - min_score)
) * 100


risk_scores = np.clip(
    risk_scores,
    0,
    100
)


# ==========================================
# RISK LEVEL
# ==========================================

def get_risk_level(score):

    if score <= 30:
        return "LOW"

    elif score <= 70:
        return "MEDIUM"

    else:
        return "HIGH"


risk_levels = [
    get_risk_level(score)
    for score in risk_scores
]


# ==========================================
# PREDICTION
# ==========================================

predictions = model.predict(
    X_scaled
)


status = [
    "ANOMALY" if p == -1 else "NORMAL"
    for p in predictions
]


# ==========================================
# CREATE RESULTS
# ==========================================

results = pd.DataFrame({

    "actual": df["label"],

    "status": status,

    "risk_score": risk_scores.round(2),

    "risk_level": risk_levels

})


# ==========================================
# SAVE
# ==========================================

results.to_csv(
    "data/risk_results.csv",
    index=False
)


# ==========================================
# DISPLAY
# ==========================================

print("\n========================================")
print("RISK SCORE SYSTEM")
print("========================================")

print(
    results.head(20).to_string(
        index=False
    )
)


print("\nRisk level distribution:")

print(
    results["risk_level"].value_counts()
)


print("\nResults saved to:")
print("data/risk_results.csv")