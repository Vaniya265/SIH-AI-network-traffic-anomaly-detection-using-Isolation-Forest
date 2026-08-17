import pandas as pd
import numpy as np
import joblib


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
# LOAD DATA
# ============================================================

print("=" * 70)
print("LOADING NSL-KDD TEST DATA")
print("=" * 70)

df = pd.read_csv(
    TEST_PATH,
    header=None,
    names=columns
)

print(f"Test dataset shape: {df.shape}")


# ============================================================
# LOAD MODELS
# ============================================================

print("\n" + "=" * 70)
print("LOADING TRAINED MODELS")
print("=" * 70)

if_model = joblib.load(IF_MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
features = joblib.load(FEATURES_PATH)
rf_model = joblib.load(RF_MODEL_PATH)

print("Isolation Forest loaded.")
print("Random Forest loaded.")


# ============================================================
# PREPARE RAW FEATURES
# ============================================================

X_raw = df.drop(
    columns=["label", "difficulty"]
)


# ============================================================
# ISOLATION FOREST PREPROCESSING
# ============================================================

print("\nPreparing Isolation Forest features...")

X_if = pd.get_dummies(
    X_raw,
    columns=[
        "protocol_type",
        "service",
        "flag"
    ]
)

X_if = X_if.reindex(
    columns=features,
    fill_value=0
)

X_if_scaled = scaler.transform(X_if)

X_if_scaled = pd.DataFrame(
    X_if_scaled,
    columns=features
)


# ============================================================
# ISOLATION FOREST PREDICTION
# ============================================================

print("Running Isolation Forest predictions...")

if_raw_predictions = if_model.predict(
    X_if_scaled
)

if_predictions = np.where(
    if_raw_predictions == -1,
    "attack",
    "normal"
)


# ============================================================
# RANDOM FOREST PREPROCESSING
# ============================================================

print("Preparing Random Forest features...")

X_rf = pd.get_dummies(
    X_raw,
    columns=[
        "protocol_type",
        "service",
        "flag"
    ]
)

X_rf = X_rf.reindex(
    columns=rf_model.feature_names_in_,
    fill_value=0
)


# ============================================================
# RANDOM FOREST PREDICTION
# ============================================================

print("Running Random Forest predictions...")

rf_raw_predictions = rf_model.predict(
    X_rf
)

rf_predictions = np.where(
    rf_raw_predictions == 1,
    "attack",
    "normal"
)


# ============================================================
# ADD PREDICTIONS
# ============================================================

df["IF_prediction"] = if_predictions
df["RF_prediction"] = rf_predictions


# ============================================================
# CONVERT TRUE LABEL TO NORMAL / ATTACK
# ============================================================

# NSL-KDD uses "normal" for normal traffic.
# Every other label represents an attack.

df["true_class"] = np.where(
    df["label"].str.lower() == "normal",
    "normal",
    "attack"
)


# ============================================================
# FIND ALL ATTACKS MISSED BY ISOLATION FOREST
# ============================================================

missed_by_if = df[
    (df["true_class"] == "attack") &
    (df["IF_prediction"] == "normal")
].copy()


print("\n" + "=" * 70)
print("ISOLATION FOREST MISSED ATTACKS")
print("=" * 70)

print(
    f"Total attack samples       : "
    f"{(df['true_class'] == 'attack').sum()}"
)

print(
    f"Attacks missed by IF       : "
    f"{len(missed_by_if)}"
)


# ============================================================
# IF OVERALL DETECTION
# ============================================================

total_attacks = (
    df["true_class"] == "attack"
).sum()

if_detected = (
    (df["true_class"] == "attack") &
    (df["IF_prediction"] == "attack")
).sum()

if_detection_rate = (
    if_detected / total_attacks * 100
)

print(
    f"IF attack detection rate  : "
    f"{if_detection_rate:.2f}%"
)


# ============================================================
# IF MISSED ATTACK BREAKDOWN
# ============================================================

if len(missed_by_if) == 0:

    print("\n🎉 Isolation Forest missed ZERO attack samples!")

else:

    print("\nAttack categories missed by Isolation Forest:")
    print("-" * 70)

    missed_summary = (
        missed_by_if
        .groupby("label")
        .size()
        .sort_values(ascending=False)
    )

    for attack_name, count in missed_summary.items():

        print(
            f"{attack_name:<25} : {count}"
        )


# ============================================================
# CHECK RANDOM FOREST ON IF-MISSED ATTACKS
# ============================================================

print("\n" + "=" * 70)
print("RANDOM FOREST ON ISOLATION FOREST MISSED ATTACKS")
print("=" * 70)

if len(missed_by_if) == 0:

    print(
        "No missed attacks exist, so RF recovery analysis "
        "is not required."
    )

else:

    rf_caught = (
        missed_by_if["RF_prediction"] == "attack"
    ).sum()

    rf_also_missed = (
        missed_by_if["RF_prediction"] == "normal"
    ).sum()

    rf_recovery_rate = (
        rf_caught / len(missed_by_if) * 100
    )

    print(
        f"IF-missed attack samples : "
        f"{len(missed_by_if)}"
    )

    print(
        f"RF recovered              : "
        f"{rf_caught}"
    )

    print(
        f"RF also missed            : "
        f"{rf_also_missed}"
    )

    print(
        f"RF recovery rate          : "
        f"{rf_recovery_rate:.2f}%"
    )


# ============================================================
# ATTACK-WISE RF RECOVERY
# ============================================================

if len(missed_by_if) > 0:

    print("\n" + "=" * 70)
    print("ATTACK-WISE RF RECOVERY")
    print("=" * 70)

    print(
        f"{'Attack':<25}"
        f"{'IF Missed':>12}"
        f"{'RF Caught':>12}"
        f"{'Recovery %':>14}"
    )

    print("-" * 70)

    for attack_name, group in missed_by_if.groupby("label"):

        total_missed = len(group)

        rf_caught = (
            group["RF_prediction"] == "attack"
        ).sum()

        recovery = (
            rf_caught / total_missed * 100
        )

        print(
            f"{attack_name:<25}"
            f"{total_missed:>12}"
            f"{rf_caught:>12}"
            f"{recovery:>13.2f}%"
        )


# ============================================================
# FIND COMPLEMENTARY CASES
# ============================================================

# These are the most interesting samples:
# Isolation Forest missed them,
# but Random Forest caught them.

complementary = missed_by_if[
    missed_by_if["RF_prediction"] == "attack"
].copy()


print("\n" + "=" * 70)
print("COMPLEMENTARY DETECTIONS")
print("=" * 70)

print(
    "These are attacks missed by Isolation Forest "
    "but detected by Random Forest."
)

print(
    f"\nTotal complementary detections: "
    f"{len(complementary)}"
)


if len(complementary) > 0:

    complementary_summary = (
        complementary
        .groupby("label")
        .size()
        .sort_values(ascending=False)
    )

    print("\nAttack categories recovered by RF:")

    for attack_name, count in complementary_summary.items():

        print(
            f"{attack_name:<25} : {count}"
        )

else:

    print(
        "\nRF did not recover any attack that "
        "Isolation Forest missed."
    )


# ============================================================
# SAVE ALL MISSED ATTACKS
# ============================================================

output_columns = [
    "label",
    "difficulty",
    "IF_prediction",
    "RF_prediction"
]

missed_by_if[output_columns].to_csv(
    "data/if_missed_attacks.csv",
    index=False
)


# ============================================================
# SAVE COMPLEMENTARY DETECTIONS
# ============================================================

complementary[output_columns].to_csv(
    "data/rf_recovered_if_misses.csv",
    index=False
)


# ============================================================
# FINAL INTERPRETATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL ANALYSIS")
print("=" * 70)

if len(missed_by_if) == 0:

    print(
        "Isolation Forest missed no attacks in this test dataset."
    )

    print(
        "Random Forest is therefore not required as a "
        "safety-net based on this experiment."
    )

elif len(complementary) > 0:

    recovery_rate = (
        len(complementary) /
        len(missed_by_if)
    ) * 100

    print(
        f"Isolation Forest missed {len(missed_by_if)} attack samples."
    )

    print(
        f"Random Forest recovered {len(complementary)} "
        f"of those samples ({recovery_rate:.2f}%)."
    )

    print(
        "\nRandom Forest provides complementary detection "
        "and may be useful as a secondary safety layer."
    )

else:

    print(
        f"Isolation Forest missed {len(missed_by_if)} attack samples."
    )

    print(
        "Random Forest did not recover any of those missed samples."
    )

    print(
        "\nNo evidence from this experiment that Random Forest "
        "improves the Isolation Forest system."
    )


# ============================================================
# OUTPUT FILES
# ============================================================

print("\n" + "=" * 70)
print("FILES SAVED")
print("=" * 70)

print(
    "data/if_missed_attacks.csv"
)

print(
    "data/rf_recovered_if_misses.csv"
)

print("\nAnalysis complete.")