"""
11_unseen_attack_evaluation.py

Person 4's core job: isolate attack types that NEVER appeared in training
(genuinely unseen), and measure how well the model catches them specifically.

This is the single most important number for the SIH1451 pitch —
proof the model catches attacks it was never trained on.
"""

import pandas as pd

# ==========================================
# 1. LOAD TRAIN AND TEST DATA (same columns as other scripts)
# ==========================================

columns = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty"
]

train = pd.read_csv("data/KDDTrain+.txt", names=columns)
test = pd.read_csv("data/KDDTest+.txt", names=columns)

# ==========================================
# 2. FIND ATTACK TYPES THAT NEVER APPEAR IN TRAINING
# ==========================================

train_labels = set(train["label"].unique())
test_labels = set(test["label"].unique())
unseen_labels = sorted(test_labels - train_labels)

print("========================================")
print("GENUINELY UNSEEN ATTACK TYPES")
print("========================================")
print(f"Found {len(unseen_labels)} attack types in test data that NEVER appeared in training:")
for label in unseen_labels:
    print(f"  - {label}")

# ==========================================
# 3. LOAD THE DETECTION RESULTS (already produced by 04_detect.py)
# ==========================================

results = pd.read_csv("data/detection_results.csv")

# ==========================================
# 4. ISOLATE JUST THE UNSEEN-ATTACK ROWS
# ==========================================

unseen_rows = results[results["actual"].isin(unseen_labels)]
caught = (unseen_rows["predicted"] == "anomaly").sum()
missed = (unseen_rows["predicted"] == "normal").sum()
total_unseen = len(unseen_rows)

catch_rate_unseen = caught / total_unseen * 100 if total_unseen > 0 else 0

print("\n========================================")
print("CATCH RATE ON GENUINELY UNSEEN ATTACKS")
print("========================================")
print(f"Total unseen-attack rows tested : {total_unseen}")
print(f"Correctly flagged (caught)      : {caught}")
print(f"Missed (wrongly marked normal)  : {missed}")
print(f"CATCH RATE ON UNSEEN ATTACKS    : {catch_rate_unseen:.1f}%")

# ==========================================
# 5. FALSE ALARM RATE (for comparison, on normal traffic)
# ==========================================

normal_rows = results[results["actual"] == "normal"]
false_alarms = (normal_rows["predicted"] == "anomaly").sum()
false_alarm_rate = false_alarms / len(normal_rows) * 100

print(f"\nFalse alarm rate (normal traffic wrongly flagged): {false_alarm_rate:.1f}%")

# ==========================================
# 6. BREAKDOWN BY EACH UNSEEN ATTACK TYPE (useful for the pitch/demo)
# ==========================================

print("\n========================================")
print("BREAKDOWN BY EACH UNSEEN ATTACK TYPE")
print("========================================")
breakdown = unseen_rows.groupby("actual")["predicted"].apply(
    lambda x: f"{(x == 'anomaly').sum()}/{len(x)} caught"
)
print(breakdown.to_string())

# ==========================================
# 7. SAVE A SMALL DEMO-READY FILE
# ==========================================

# Pick unseen-attack rows that were CORRECTLY caught — safe to use in a live demo
confirmed_catches = unseen_rows[unseen_rows["predicted"] == "anomaly"]
demo_sample = confirmed_catches.sample(min(5, len(confirmed_catches)), random_state=42)
demo_sample.to_csv("data/demo_unseen_attack_confirmed.csv", index=False)

print("\n========================================")
print("DEMO FILE SAVED")
print("========================================")
print("Saved 5 CONFIRMED unseen-attack rows (that got correctly flagged) to:")
print("  data/demo_unseen_attack_confirmed.csv")
print("Use these specific rows in your live demo — they are guaranteed to show up as flagged.")
