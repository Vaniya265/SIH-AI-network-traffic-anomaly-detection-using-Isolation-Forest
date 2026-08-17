import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import confusion_matrix


# ============================================================
# PATHS
# ============================================================

TEST_PATH = "data/KDDTest+.txt"

IF_MODEL_PATH = "models/isolation_forest_improved.pkl"
SCALER_PATH = "models/scaler.pkl"
FEATURES_PATH = "models/features.pkl"

RF_MODEL_PATH = "models/random_forest.pkl"


# ============================================================
# NSL-KDD COLUMN NAMES
# ============================================================

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


# ============================================================
# LOAD TEST DATA
# ============================================================

print("Loading NSL-KDD test data...")

df = pd.read_csv(
    TEST_PATH,
    header=None,
    names=columns
)

print("Test dataset shape:", df.shape)


# ============================================================
# LOAD MODELS
# ============================================================

print("\nLoading models...")

if_model = joblib.load(IF_MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
features = joblib.load(FEATURES_PATH)

rf_model = joblib.load(RF_MODEL_PATH)

print("Isolation Forest loaded.")
print("Random Forest loaded.")


# ============================================================
# COMMON FEATURES
# ============================================================

X_raw = df.drop(columns=["label", "difficulty"])


# ============================================================
# 1. ISOLATION FOREST PREPROCESSING
# ============================================================

print("\nPreparing Isolation Forest features...")

X_if = pd.get_dummies(
    X_raw,
    columns=["protocol_type", "service", "flag"]
)

# Make sure exactly the same columns used during training exist
X_if = X_if.reindex(
    columns=features,
    fill_value=0
)

# Apply the same scaler used during preprocessing
X_if_scaled = scaler.transform(X_if)

# Convert to DataFrame to preserve feature names
X_if_scaled = pd.DataFrame(
    X_if_scaled,
    columns=features
)


# ============================================================
# ISOLATION FOREST PREDICTION
# ============================================================

print("Running Isolation Forest predictions...")

if_raw_predictions = if_model.predict(X_if_scaled)

# Isolation Forest:
# -1 = anomaly
#  1 = normal

if_predictions = np.where(
    if_raw_predictions == -1,
    "attack",
    "normal"
)


# ============================================================
# 2. RANDOM FOREST PREPROCESSING
# ============================================================

print("Preparing Random Forest features...")

X_rf = pd.get_dummies(
    X_raw,
    columns=["protocol_type", "service", "flag"]
)

# Make test features identical to RF training features
X_rf = X_rf.reindex(
    columns=rf_model.feature_names_in_,
    fill_value=0
)


# ============================================================
# RANDOM FOREST PREDICTION
# ============================================================

print("Running Random Forest predictions...")

rf_raw_predictions = rf_model.predict(X_rf)

# RF:
# 0 = normal
# 1 = attack

rf_predictions = np.where(
    rf_raw_predictions == 1,
    "attack",
    "normal"
)


# ============================================================
# ADD PREDICTIONS TO DATAFRAME
# ============================================================

df["IF_prediction"] = if_predictions
df["RF_prediction"] = rf_predictions


# ============================================================
# ATTACKS WE WANT TO INVESTIGATE
# ============================================================

target_attacks = [
    "nmap",
    "warezclient"
]


# ============================================================
# FUNCTION TO ANALYZE AN ATTACK
# ============================================================

def analyze_attack(attack_name):

    attack_data = df[
        df["label"].str.lower() == attack_name.lower()
    ]

    total = len(attack_data)

    if total == 0:
        print(f"\nNo samples found for {attack_name}")
        return

    # Isolation Forest
    if_detected = (
        attack_data["IF_prediction"] == "attack"
    ).sum()

    if_missed = total - if_detected

    # Random Forest
    rf_detected = (
        attack_data["RF_prediction"] == "attack"
    ).sum()

    rf_missed = total - rf_detected

    # Percentages
    if_detection_rate = (
        if_detected / total
    ) * 100

    if_miss_rate = (
        if_missed / total
    ) * 100

    rf_detection_rate = (
        rf_detected / total
    ) * 100

    rf_miss_rate = (
        rf_missed / total
    ) * 100

    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    print("\n" + "=" * 60)
    print(f"ATTACK ANALYSIS: {attack_name.upper()}")
    print("=" * 60)

    print(f"Total samples             : {total}")

    print("\n--- Isolation Forest ---")
    print(f"Detected                  : {if_detected}")
    print(f"Missed                    : {if_missed}")
    print(f"Detection Rate            : {if_detection_rate:.2f}%")
    print(f"Miss Rate                 : {if_miss_rate:.2f}%")

    print("\n--- Random Forest ---")
    print(f"Detected                  : {rf_detected}")
    print(f"Missed                    : {rf_missed}")
    print(f"Detection Rate            : {rf_detection_rate:.2f}%")
    print(f"Miss Rate                 : {rf_miss_rate:.2f}%")

    # Improvement
    improvement = (
        rf_detection_rate - if_detection_rate
    )

    print("\n--- Comparison ---")

    if improvement > 0:
        print(
            f"Random Forest improvement : +{improvement:.2f} percentage points"
        )

    elif improvement < 0:
        print(
            f"Random Forest change      : {improvement:.2f} percentage points"
        )

    else:
        print(
            "Random Forest improvement : 0.00 percentage points"
        )


# ============================================================
# RUN ANALYSIS
# ============================================================

for attack in target_attacks:
    analyze_attack(attack)


# ============================================================
# OVERALL TWO-ATTACK COMPARISON
# ============================================================

selected = df[
    df["label"].str.lower().isin(
        [x.lower() for x in target_attacks]
    )
]

total_selected = len(selected)

if_detected_total = (
    selected["IF_prediction"] == "attack"
).sum()

rf_detected_total = (
    selected["RF_prediction"] == "attack"
).sum()

if_rate_total = (
    if_detected_total / total_selected
) * 100

rf_rate_total = (
    rf_detected_total / total_selected
) * 100


print("\n" + "=" * 60)
print("OVERALL NMAP + WAREZCLIENT COMPARISON")
print("=" * 60)

print(f"Total attack samples       : {total_selected}")

print(
    f"Isolation Forest detected  : "
    f"{if_detected_total} ({if_rate_total:.2f}%)"
)

print(
    f"Random Forest detected     : "
    f"{rf_detected_total} ({rf_rate_total:.2f}%)"
)

print(
    f"\nDifference                 : "
    f"{rf_rate_total - if_rate_total:+.2f} percentage points"
)


# ============================================================
# SAVE COMPARISON RESULTS
# ============================================================

comparison_columns = [
    "label",
    "IF_prediction",
    "RF_prediction"
]

comparison_data = df[
    df["label"].str.lower().isin(
        [x.lower() for x in target_attacks]
    )
][comparison_columns]

comparison_data.to_csv(
    "data/attack_comparison.csv",
    index=False
)

print("\nDetailed results saved to:")
print("data/attack_comparison.csv")