"""
13_live_judge_demo.py

Lets a judge pick ANY row number from the test set live, and runs it
through the real API — proving the model isn't just working on
pre-picked, rehearsed examples.

Usage: run this, then type a row number when asked (0 to 22543),
or type 'random' to let it pick one for you.
"""

import pandas as pd
import requests
import os
import random

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) \
    if os.path.basename(os.path.dirname(os.path.abspath(__file__))) == "src" \
    else os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

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

test = pd.read_csv(os.path.join(DATA_DIR, "KDDTest+.txt"), names=columns)

API_URL = "http://127.0.0.1:8000/predict"

print("=" * 60)
print("LIVE JUDGE DEMO — pick any row, unscripted")
print("=" * 60)
print(f"There are {len(test)} rows available (numbered 0 to {len(test)-1}).")
print("Ask the judge to call out any number, or type 'random'.\n")

while True:
    choice = input("Enter a row number (or 'random', or 'quit'): ").strip()

    if choice.lower() == "quit":
        break

    if choice.lower() == "random":
        row_index = random.randint(0, len(test) - 1)
    else:
        try:
            row_index = int(choice)
            if row_index < 0 or row_index >= len(test):
                print(f"Please enter a number between 0 and {len(test)-1}.\n")
                continue
        except ValueError:
            print("Please enter a valid number, 'random', or 'quit'.\n")
            continue

    row = test.iloc[row_index]
    actual_label = row["label"]
    payload = row.drop(labels=["label", "difficulty"]).to_dict()

    try:
        response = requests.post(API_URL, json=payload, timeout=5)
        result = response.json()

        print(f"\n--- Row #{row_index} ---")
        print(f"  (Actual traffic type, hidden from model: {actual_label})")
        print(f"  Status      : {result.get('status')}")
        print(f"  Risk Score  : {result.get('risk_score')}")
        print(f"  Risk Level  : {result.get('risk_level')}")
        print(f"  Reasons     : {result.get('reasons')}")
        print()

    except requests.exceptions.ConnectionError:
        print("\n❌ Could not connect to the API. Is uvicorn running?\n")
        break
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
