import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ==============================
# PATHS
# ==============================

TEST_PATH = "data/KDDTest+.txt"
MODEL_PATH = "models/random_forest.pkl"


# ==============================
# COLUMN NAMES
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
# LOAD TEST DATA
# ==============================

print("Loading NSL-KDD test data...")

df = pd.read_csv(
    TEST_PATH,
    header=None,
    names=columns
)

print("Test dataset shape:", df.shape)


# ==============================
# CREATE BINARY TARGET
# ==============================

df["target"] = (df["label"] != "normal").astype(int)

X_test = df.drop(
    columns=["label", "difficulty", "target"]
)

y_test = df["target"]


# ==============================
# ENCODE CATEGORICAL FEATURES
# ==============================

categorical_columns = [
    "protocol_type",
    "service",
    "flag"
]

X_test = pd.get_dummies(
    X_test,
    columns=categorical_columns
)


# ==============================
# LOAD MODEL
# ==============================

model = joblib.load(MODEL_PATH)


# ==============================
# ALIGN FEATURES
# ==============================

# Make sure test columns match
# the columns expected by the model.

X_test = X_test.reindex(
    columns=model.feature_names_in_,
    fill_value=0
)


# ==============================
# PREDICTION
# ==============================

predictions = model.predict(X_test)


# ==============================
# METRICS
# ==============================

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

cm = confusion_matrix(
    y_test,
    predictions
)


# ==============================
# RESULTS
# ==============================

print("\n========================================")
print("RANDOM FOREST TEST PERFORMANCE")
print("========================================")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")


print("\n========================================")
print("CONFUSION MATRIX")
print("========================================")

print(cm)


print("\n========================================")
print("CLASSIFICATION REPORT")
print("========================================")

print(
    classification_report(
        y_test,
        predictions,
        target_names=["normal", "attack"]
    )
)