"""Prediction routes for the trained diabetes and heart-risk models."""

import os
import joblib
import pandas as pd
from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required

from utils.validators import (
    DIABETES_FIELDS,
    DIABETES_BOUNDS,
    HEART_FIELDS,
    HEART_BOUNDS,
    validate_fields,
)


predict_bp = Blueprint("predict", __name__)
BASE = os.path.abspath(
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "saved_models")
)
OPT_HEART_PIPELINE_PATH = os.path.join(BASE, "heart_optimized", "heart_optimized_pipeline.joblib")

try:
    D_MODEL = joblib.load(f"{BASE}/diabetes_model.pkl")
    D_SCALER = joblib.load(f"{BASE}/diabetes_scaler.pkl")
    H_MODEL = joblib.load(f"{BASE}/heart_model.pkl")
    H_SCALER = joblib.load(f"{BASE}/heart_scaler.pkl")
    H_OPT_MODEL = None
    if os.path.exists(OPT_HEART_PIPELINE_PATH):
        try:
            H_OPT_MODEL = joblib.load(OPT_HEART_PIPELINE_PATH)
        except Exception as exc:
            # This artifact is optional; skip it if it was serialized by an
            # incompatible scikit-learn version and keep the baseline model.
            print(f"Warning: optimized heart pipeline unavailable: {exc}")
    print("Models loaded")
except FileNotFoundError as e:
    print(f"Warning: {e}")
    D_MODEL = D_SCALER = H_MODEL = H_SCALER = H_OPT_MODEL = None


def _advice(disease, prob):
    high = prob >= 0.5
    risk = "High Risk" if high else "Low Risk"
    if disease == "diabetes":
        advice = (
            "Elevated diabetes risk. Consult endocrinologist. Monitor HbA1c and glucose regularly."
            if high
            else "Low diabetes risk. Maintain healthy lifestyle and annual glucose checks."
        )
    else:
        advice = (
            "Elevated cardiovascular risk. Seek cardiology consultation. Monitor blood pressure daily."
            if high
            else "Low cardiovascular risk. Continue exercise and a heart-healthy diet."
        )
    return risk, advice


def _engineer_heart_features(data):
    row = {
        "age_years": float(data["age_years"]),
        "gender": float(data["gender"]),
        "height": float(data["height"]),
        "weight": float(data["weight"]),
        "bmi": float(data["bmi"]),
        "ap_hi": float(data["ap_hi"]),
        "ap_lo": float(data["ap_lo"]),
        "cholesterol": float(data["cholesterol"]),
        "gluc": float(data["gluc"]),
        "smoke": float(data["smoke"]),
        "alco": float(data["alco"]),
        "active": float(data["active"]),
    }
    frame = pd.DataFrame([row])
    frame["pulse_pressure"] = frame["ap_hi"] - frame["ap_lo"]
    frame["mean_arterial_pressure"] = (frame["ap_hi"] + 2 * frame["ap_lo"]) / 3
    frame["bp_ratio"] = frame["ap_hi"] / frame["ap_lo"].clip(lower=1)
    frame["chol_gluc_risk"] = frame["cholesterol"] * frame["gluc"]
    frame["age_bp_ratio"] = frame["age_years"] / frame["ap_hi"].clip(lower=1)
    frame["age_chol_ratio"] = frame["age_years"] / frame["cholesterol"].clip(lower=1)
    frame["weight_height_ratio"] = frame["weight"] / frame["height"].clip(lower=1)
    frame["activity_smoke_interaction"] = (1 - frame["active"]) * frame["smoke"]
    frame["age_group"] = pd.cut(
        frame["age_years"], bins=[0, 40, 55, 120], labels=False, include_lowest=True
    ).fillna(1).astype(int)
    frame["bmi_group"] = pd.cut(
        frame["bmi"], bins=[0, 18.5, 25, 30, 100], labels=False, include_lowest=True
    ).fillna(1).astype(int)
    return frame


@predict_bp.route("/predict/diabetes", methods=["POST"])
@jwt_required()
def predict_diabetes():
    if D_MODEL is None:
        return jsonify({"error": "Run train_models.py first"}), 503

    data = request.get_json(silent=True) or {}
    vals, err = validate_fields(data, DIABETES_FIELDS, DIABETES_BOUNDS)
    if err:
        return jsonify({"error": err}), 400

    prob = float(D_MODEL.predict_proba(D_SCALER.transform([vals]))[0][1])
    risk, advice = _advice("diabetes", prob)
    return jsonify(
        {
            "prediction": int(prob >= 0.5),
            "probability": round(prob, 4),
            "risk_level": risk,
            "advice": advice,
        }
    )


@predict_bp.route("/predict/heart", methods=["POST"])
@jwt_required()
def predict_heart():
    if H_MODEL is None and H_OPT_MODEL is None:
        return jsonify({"error": "Run train_models.py first"}), 503

    data = request.get_json(silent=True) or {}
    if "bmi" not in data and "weight" in data and "height" in data:
        try:
            data["bmi"] = round(float(data["weight"]) / (float(data["height"]) / 100) ** 2, 1)
        except Exception:
            pass

    vals, err = validate_fields(data, HEART_FIELDS, HEART_BOUNDS)
    if err:
        return jsonify({"error": err}), 400

    if H_OPT_MODEL is not None:
        validated = dict(zip(HEART_FIELDS, vals))
        engineered = _engineer_heart_features(validated)
        prob = float(H_OPT_MODEL.predict_proba(engineered)[0][1])
        model_name = "Optimized Heart Pipeline"
    else:
        prob = float(H_MODEL.predict_proba(H_SCALER.transform([vals]))[0][1])
        model_name = "Baseline Heart Model"
    risk, advice = _advice("heart", prob)
    return jsonify(
        {
            "prediction": int(prob >= 0.5),
            "probability": round(prob, 4),
            "risk_level": risk,
            "advice": advice,
            "model_used": model_name,
        }
    )
