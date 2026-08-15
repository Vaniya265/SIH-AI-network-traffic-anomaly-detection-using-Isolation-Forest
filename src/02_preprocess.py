import pandas as pd
from sklearn.preprocessing import StandardScaler
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
# 2. LOAD DATASET
# ==========================================

df = pd.read_csv(
    "data/KDDTrain+.txt",
    names=columns
)

print("Original dataset shape:", df.shape)


# ==========================================
# 3. KEEP ONLY NORMAL TRAFFIC
# ==========================================

normal_df = df[df["label"] == "normal"].copy()

print("Normal traffic shape:", normal_df.shape)


# ==========================================
# 4. REMOVE LABEL + DIFFICULTY
# ==========================================

X = normal_df.drop(
    columns=["label", "difficulty"]
)


# ==========================================
# 5. ONE-HOT ENCODE CATEGORICAL FEATURES
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
# 6. CONVERT BOOLEAN → INTEGER
# ==========================================

X = X.astype(float)


# ==========================================
# 7. SCALE FEATURES
# ==========================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# ==========================================
# 8. SAVE SCALER + FEATURE NAMES
# ==========================================

joblib.dump(
    scaler,
    "models/scaler.pkl"
)

joblib.dump(
    X.columns.tolist(),
    "models/features.pkl"
)


# ==========================================
# 9. SAVE PROCESSED NORMAL DATA
# ==========================================

processed_df = pd.DataFrame(
    X_scaled,
    columns=X.columns
)

processed_df.to_csv(
    "data/normal_processed.csv",
    index=False
)


# ==========================================
# 10. FINAL INFORMATION
# ==========================================

print("\nPreprocessing completed!")

print("Processed shape:", processed_df.shape)

print("Scaler saved to: models/scaler.pkl")

print("Feature list saved to: models/features.pkl")

print("Processed data saved to: data/normal_processed.csv")