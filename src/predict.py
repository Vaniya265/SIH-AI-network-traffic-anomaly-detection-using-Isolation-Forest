import os
import pandas as pd
import numpy as np
import joblib


# ==========================================
# LOAD MODEL + PREPROCESSING
# ==========================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "isolation_forest_improved.pkl"
)

SCALER_PATH = os.path.join(
    MODEL_DIR,
    "scaler.pkl"
)

FEATURES_PATH = os.path.join(
    MODEL_DIR,
    "features.pkl"
)

CALIBRATION_PATH = os.path.join(
    MODEL_DIR,
    "risk_calibration.pkl"
)


model = joblib.load(MODEL_PATH)

scaler = joblib.load(SCALER_PATH)

feature_names = joblib.load(FEATURES_PATH)

calibration = joblib.load(CALIBRATION_PATH)


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


# ==========================================
# PREPROCESS
# ==========================================

def preprocess_input(data):

    df = pd.DataFrame([data])

    categorical_columns = [
        "protocol_type",
        "service",
        "flag"
    ]

    df = pd.get_dummies(
        df,
        columns=categorical_columns
    )

    df = df.reindex(
        columns=feature_names,
        fill_value=0
    )

    scaled = scaler.transform(df)

    scaled = pd.DataFrame(
        scaled,
        columns=feature_names
    )

    return scaled


# ==========================================
# CALCULATE RISK SCORE
# ==========================================

def calculate_risk_score(raw_score):

    p95 = calibration["p95"]
    p99 = calibration["p99"]

    # --------------------------------------
    # LOW RISK
    # --------------------------------------

    if raw_score <= p95:

        return 20.0


    # --------------------------------------
    # MEDIUM RISK
    # --------------------------------------

    elif raw_score <= p99:

        ratio = (
            (raw_score - p95)
            /
            (p99 - p95)
        )

        score = 20 + (ratio * 50)

        return round(
            float(np.clip(score, 20, 70)),
            2
        )


    # --------------------------------------
    # HIGH RISK
    # --------------------------------------

    else:

        # Instead of immediately jumping to 100,
        # use a smooth logarithmic scale.

        excess = (
            raw_score - p99
        )

        score = (
            70
            +
            30
            *
            (
                1
                -
                np.exp(-5 * excess)
            )
        )

        return round(
            float(np.clip(score, 70, 100)),
            2
        )

# ==========================================
# PREDICT ONE TRAFFIC RECORD
# ==========================================

def predict_one(data):

    X = preprocess_input(data)


    # --------------------------------------
    # MODEL PREDICTION
    # --------------------------------------

    prediction = model.predict(X)[0]


    # --------------------------------------
    # ANOMALY SCORE
    # --------------------------------------

    raw_score = -model.decision_function(X)[0]


    # --------------------------------------
    # RISK SCORE
    # --------------------------------------

    risk_score = calculate_risk_score(
        raw_score
    )


    # --------------------------------------
    # STATUS
    # --------------------------------------

    status = (
        "ANOMALY"
        if prediction == -1
        else "NORMAL"
    )


    # --------------------------------------
    # RISK LEVEL
    # --------------------------------------

    risk_level = get_risk_level(
        risk_score
    )


    return {

        "status": status,

        "risk_score": risk_score,

        "risk_level": risk_level

    }