"""
Run this AFTER starting the server, to confirm everything works using
the team's actual demo files (demo_normal.csv, demo_known_attack.csv,
demo_hidden_attack.csv) instead of made-up data.

USAGE:
    1. cd into the src/ folder
    2. uvicorn api:app --reload --port 8000
    3. In a second terminal, from the repo root: python test_live_api.py
"""

import requests
import pandas as pd

BASE = "http://localhost:8000"


def rows_from_csv(path, n=None):
    df = pd.read_csv(path)
    if n:
        df = df.head(n)
    rows = df.drop(columns=["label", "difficulty"], errors="ignore")
    return rows.to_dict(orient="records")


print("1. Health check:")
print(requests.get(f"{BASE}/health").json())

print("\n2. Normal traffic (first 3 rows) -> expect mostly Safe/NORMAL:")
for row in rows_from_csv("demo_normal.csv", n=3):
    r = requests.post(f"{BASE}/predict", json=row).json()
    print(f"   {r['status']:8s} risk={r['risk_score']:>6} ({r['risk_level']})  reasons={r['reasons'][:1]}")

print("\n3. Known attack (first 3 rows) -> expect mostly ANOMALY:")
for row in rows_from_csv("demo_known_attack.csv", n=3):
    r = requests.post(f"{BASE}/predict", json=row).json()
    print(f"   {r['status']:8s} risk={r['risk_score']:>6} ({r['risk_level']})  reasons={r['reasons'][:1]}")

print("\n4. HIDDEN/unseen attack -> the centerpiece demo moment, expect ANOMALY:")
for row in rows_from_csv("demo_hidden_attack.csv"):
    r = requests.post(f"{BASE}/predict", json=row).json()
    print(f"   {r['status']:8s} risk={r['risk_score']:>6} ({r['risk_level']})  reasons={r['reasons']}")

print("\nIf the hidden attack row(s) above show ANOMALY, your centerpiece demo moment works end to end.")
print("Hand the URL http://localhost:8000/predict (or your deployed URL) to Person 6.")
