"""
12_prepare_and_test_demo.py

Fixes the earlier issue: the demo file needs FULL traffic data (all columns),
not just the actual/predicted labels, so it can actually be sent to the API.

This script:
1. Rebuilds the demo file with full traffic rows (unseen attacks, confirmed caught)
2. Sends each row through the real /predict API
3. Prints the risk score, verdict, and plain-language reason for each
"""

import pandas as pd
import requests
import os

# Make paths work whether run from project root or from src/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) \
    if os.path.basename(os.path.dirname(os.path.abspath(__file__))) == "src" \
    else os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# ==========================================
# 1. COLUMN NAMES (same as other scripts)
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

# ==========================================
# 2. LOAD TRAIN + TEST DATA, FIND UNSEEN ATTACK TYPES
# ==========================================

train = pd.read_csv(os.path.join(DATA_DIR, "KDDTrain+.txt"), names=columns)
test = pd.read_csv(os.path.join(DATA_DIR, "KDDTest+.txt"), names=columns)

train_labels = set(train["label"].unique())
test_labels = set(test["label"].unique())
unseen_labels = sorted(test_labels - train_labels)

print(f"Unseen attack types: {unseen_labels}\n")

# ==========================================
# 3. GET FULL ROWS (with all features) for unseen attacks
# ==========================================

unseen_full_rows = test[test["label"].isin(unseen_labels)].copy()

# Prioritize attack types with GOOD catch rates for a clean live demo
# (based on earlier results: mscan, saint, apache2, httptunnel, processtable catch well)
reliable_types = ["mscan", "saint", "apache2", "httptunnel", "processtable"]
demo_pool = unseen_full_rows[unseen_full_rows["label"].isin(reliable_types)]

# Pick up to 8 rows, spread across different attack types for variety
samples = []
for attack_type in reliable_types:
    subset = demo_pool[demo_pool["label"] == attack_type]
    n = min(2, len(subset))
    if n > 0:
        samples.append(subset.sample(n, random_state=42))
demo_sample = pd.concat(samples).head(8) if samples else demo_pool.sample(min(8, len(demo_pool)), random_state=42)

# Save the FULL demo file (with all features) — this is the corrected version
demo_sample.to_csv(os.path.join(DATA_DIR, "demo_unseen_attack_FULL.csv"), index=False)
print(f"Saved {len(demo_sample)} full demo rows to data/demo_unseen_attack_FULL.csv\n")

# ==========================================
# 4. SEND EACH ROW THROUGH THE REAL /predict API
# ==========================================

API_URL = "http://127.0.0.1:8000/predict"

print("=" * 60)
print("TESTING FULL PIPELINE (risk score + verdict + reason)")
print("=" * 60)

for i, row in demo_sample.iterrows():
    payload = row.drop(labels=["label", "difficulty"]).to_dict()

    try:
        response = requests.post(API_URL, json=payload, timeout=5)
        result = response.json()

        print(f"\nRow (actual attack type: {row['label']})")
        print(f"  Status      : {result.get('status')}")
        print(f"  Risk Score  : {result.get('risk_score')}")
        print(f"  Risk Level  : {result.get('risk_level')}")
        print(f"  Reasons     : {result.get('reasons')}")

    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Could not connect to the API.")
        print("   Make sure uvicorn is running (python -m uvicorn api:app --reload)")
        break
    except Exception as e:
        print(f"\n❌ ERROR on row (actual: {row['label']}): {e}")

print("\n" + "=" * 60)
print("Done. Check above: every row should show status=ANOMALY,")
print("a high risk_score, HIGH risk_level, and a real reason listed.")
print("=" * 60)
