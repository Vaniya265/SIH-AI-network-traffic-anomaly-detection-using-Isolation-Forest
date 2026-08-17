import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ==============================
# PATHS
# ==============================

DATA_PATH = "data/KDDTrain+.txt"
MODEL_PATH = "models/random_forest.pkl"
FEATURES_PATH = "models/features.pkl"


# ==============================
# NSL-KDD COLUMN NAMES
# ==============================

columns = [
    "duration", "protocol_type", "service", "flag",
    "src_bytes", "dst_bytes", "land", "wrong_fragment",
    "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted",
    "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "label", "difficulty"
]


# ==============================
# LOAD DATA
# ==============================

print("Loading NSL-KDD training data...")

df = pd.read_csv(
    DATA_PATH,
    header=None,
    names=columns
)

print("Original dataset shape:", df.shape)


# ==============================
# CREATE BINARY TARGET
# ==============================

df["target"] = (df["label"] != "normal").astype(int)

print("\nTarget distribution:")
print(df["target"].value_counts())

print("\n0 = Normal")
print("1 = Attack")


# ==============================
# DROP UNUSED COLUMNS
# ==============================

X = df.drop(columns=["label", "difficulty", "target"])
y = df["target"]


# ==============================
# ONE-HOT ENCODE CATEGORICAL DATA
# ==============================

categorical_columns = [
    "protocol_type",
    "service",
    "flag"
]

X = pd.get_dummies(
    X,
    columns=categorical_columns
)

print("\nEncoded feature shape:", X.shape)


# ==============================
# TRAIN RANDOM FOREST
# ==============================

print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

model.fit(X, y)


# ==============================
# TRAINING PERFORMANCE
# ==============================

predictions = model.predict(X)

accuracy = accuracy_score(y, predictions)
precision = precision_score(y, predictions)
recall = recall_score(y, predictions)
f1 = f1_score(y, predictions)

print("\n========================================")
print("RANDOM FOREST TRAINING PERFORMANCE")
print("========================================")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")


# ==============================
# SAVE MODEL
# ==============================

joblib.dump(model, MODEL_PATH)

print("\nRandom Forest trained successfully!")
print("Model saved to:", MODEL_PATH)