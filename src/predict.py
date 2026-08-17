import os
import pandas as pd
import numpy as np
import joblib

from feature_deviation import FeatureDeviationEngine


# ==========================================
# PATHS
# ==========================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

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

RF_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "random_forest.pkl"
)


# ==========================================
# LOAD MODELS
# ==========================================

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_names = joblib.load(FEATURES_PATH)
calibration = joblib.load(CALIBRATION_PATH)
rf_model = joblib.load(RF_MODEL_PATH)


# ==========================================
# FEATURE DEVIATION ENGINE
# ==========================================

deviation_engine = FeatureDeviationEngine(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "baseline_stats.json"
    )
)


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
# PREPROCESS FOR ISOLATION FOREST
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
# PREPROCESS FOR RANDOM FOREST
# ==========================================

def preprocess_for_rf(data):

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
        columns=rf_model.feature_names_in_,
        fill_value=0
    )

    return df


# ==========================================
# CALCULATE RISK SCORE
# ==========================================

def calculate_risk_score(raw_score):

    p95 = calibration["p95"]
    p99 = calibration["p99"]

    if raw_score <= p95:
        return 20.0

    elif raw_score <= p99:

        ratio = (
            (raw_score - p95)
            /
            (p99 - p95)
        )

        score = 20 + (ratio * 50)

        return round(
            float(
                np.clip(score, 20, 70)
            ),
            2
        )

    else:

        excess = raw_score - p99

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
            float(
                np.clip(score, 70, 100)
            ),
            2
        )


# ==========================================
# PREDICT ONE TRAFFIC RECORD
# ==========================================

def predict_one(data):

    # ======================================
    # ISOLATION FOREST
    # ======================================

    X_if = preprocess_input(data)

    prediction = model.predict(X_if)[0]

    raw_score = -model.decision_function(X_if)[0]

    risk_score = calculate_risk_score(raw_score)

    status = (
        "ANOMALY"
        if prediction == -1
        else "NORMAL"
    )

    risk_level = get_risk_level(risk_score)


    # ======================================
    # DEFAULT DETECTION SOURCE
    # ======================================

    detection_source = (
        "Isolation Forest"
        if prediction == -1
        else "None"
    )

    rf_prediction = None


    # ======================================
    # RANDOM FOREST SAFETY NET
    # ======================================

    if prediction == 1:

        X_rf = preprocess_for_rf(data)

        rf_raw_prediction = rf_model.predict(X_rf)[0]

        rf_prediction = (
            "ATTACK"
            if rf_raw_prediction == 1
            else "NORMAL"
        )

        if rf_raw_prediction == 1:

            status = "ANOMALY"

            risk_score = max(
                risk_score,
                70.0
            )

            risk_level = "HIGH"

            detection_source = (
                "Random Forest Safety Net"
            )


    # ======================================
    # WHY FLAGGED
    # ======================================

    reasons = deviation_engine.get_reasons(
        data,
        top_n=3
    )


    # ======================================
    # FINAL VERDICT
    # ======================================

    if status == "ANOMALY":

        if risk_score >= 70:
            verdict = "COMPROMISED"
        else:
            verdict = "SUSPICIOUS"

    else:

        verdict = "SAFE"


    # ======================================
    # RETURN RESULT
    # ======================================

    return {
        "status": status,
        "verdict": verdict,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "detection_source": detection_source,
        "rf_prediction": rf_prediction,
        "reasons": reasons
    }