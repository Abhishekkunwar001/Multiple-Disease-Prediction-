"""
Train diabetes and heart-risk models from the project datasets.

Outputs:
- ../saved_models/diabetes_model.pkl
- ../saved_models/diabetes_scaler.pkl
- ../saved_models/heart_model.pkl
- ../saved_models/heart_scaler.pkl
- ../saved_models/metrics.json
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier

    HAS_XGB = True
except Exception:
    HAS_XGB = False
    print("XGBoost not available")


BASE = os.path.dirname(os.path.abspath(__file__))
SAVE = os.path.join(BASE, "..", "saved_models")
DATA = os.path.join(BASE, "..", "datasets")
os.makedirs(SAVE, exist_ok=True)

DIABETES_FEATURE_LABELS = {
    "gender": "Gender",
    "age": "Age",
    "hypertension": "Hypertension",
    "heart_disease": "Heart Disease",
    "smoking_history": "Smoking History",
    "bmi": "BMI",
    "HbA1c_level": "HbA1c Level",
    "blood_glucose_level": "Blood Glucose",
}

HEART_FEATURE_LABELS = {
    "age_years": "Age (years)",
    "gender": "Gender",
    "height": "Height (cm)",
    "weight": "Weight (kg)",
    "bmi": "BMI",
    "ap_hi": "Systolic BP",
    "ap_lo": "Diastolic BP",
    "cholesterol": "Cholesterol",
    "gluc": "Glucose",
    "smoke": "Smoker",
    "alco": "Alcohol",
    "active": "Physically Active",
}


def get_metrics(model, x_test, y_test, features):
    preds = model.predict(x_test)
    feature_importance = {}
    if hasattr(model, "feature_importances_"):
        feature_importance = {
            key: round(float(value), 4)
            for key, value in zip(features, model.feature_importances_)
        }
    elif hasattr(model, "coef_"):
        feature_importance = {
            key: round(float(abs(value)), 4)
            for key, value in zip(features, model.coef_[0])
        }

    return {
        "accuracy": round(accuracy_score(y_test, preds) * 100, 2),
        "precision": round(precision_score(y_test, preds, zero_division=0) * 100, 2),
        "recall": round(recall_score(y_test, preds, zero_division=0) * 100, 2),
        "f1": round(f1_score(y_test, preds, zero_division=0) * 100, 2),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
        "feature_importance": feature_importance,
    }


def train_pool(x_train, x_test, y_train, y_test, features):
    pool = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=12, random_state=42, n_jobs=1
        ),
    }
    if HAS_XGB:
        pool["XGBoost"] = XGBClassifier(
            n_estimators=200, random_state=42, eval_metric="logloss"
        )

    results = {}
    best_acc = 0
    best_model = None
    best_name = ""
    trained_models = {}

    for name, model in pool.items():
        model.fit(x_train, y_train)
        trained_models[name] = model
        metrics = get_metrics(model, x_test, y_test, features)
        results[name] = metrics
        print(f"  {name}: acc={metrics['accuracy']}% f1={metrics['f1']}%")
        if metrics["accuracy"] > best_acc:
            best_acc = metrics["accuracy"]
            best_model = model
            best_name = name

    results["best_model"] = best_name
    results["best_accuracy"] = best_acc
    results["features"] = features
    return results, trained_models, best_model
print("\n== DIABETES ==")
df_diabetes = pd.read_csv(os.path.join(DATA, "diabetes_new.csv"))
smoking_map = {"never": 0, "No Info": 1, "current": 2, "former": 3, "ever": 4, "not current": 5}
gender_map = {"Female": 0, "Male": 1, "Other": 2}

df_diabetes["smoking_history"] = df_diabetes["smoking_history"].map(smoking_map).fillna(0).astype(int)
df_diabetes["gender"] = df_diabetes["gender"].map(gender_map).fillna(0).astype(int)

diabetes_features = [column for column in df_diabetes.columns if column != "diabetes"]
x_diabetes = df_diabetes[diabetes_features].values.astype(float)
y_diabetes = df_diabetes["diabetes"].values

positive_idx = np.where(y_diabetes == 1)[0]
negative_idx = np.where(y_diabetes == 0)[0]
np.random.seed(42)
negative_sample = np.random.choice(negative_idx, len(positive_idx) * 3, replace=False)
selected_idx = np.concatenate([positive_idx, negative_sample])
np.random.shuffle(selected_idx)
x_diabetes = x_diabetes[selected_idx]
y_diabetes = y_diabetes[selected_idx]

diabetes_scaler = StandardScaler()
x_diabetes_scaled = diabetes_scaler.fit_transform(x_diabetes)
x_train_d, x_test_d, y_train_d, y_test_d = train_test_split(
    x_diabetes_scaled, y_diabetes, test_size=0.2, random_state=42, stratify=y_diabetes
)

diabetes_results, diabetes_models, best_diabetes_model = train_pool(
    x_train_d, x_test_d, y_train_d, y_test_d, diabetes_features
)
diabetes_results["dataset"] = {
    "total": int(len(df_diabetes)),
    "features": len(diabetes_features),
    "train": int(len(x_train_d)),
    "test": int(len(x_test_d)),
}
diabetes_results["encoders"] = {
    "smoking_history": {str(v): k for k, v in smoking_map.items()},
    "gender": {str(v): k for k, v in gender_map.items()},
}
diabetes_results["feature_labels"] = DIABETES_FEATURE_LABELS

joblib.dump(best_diabetes_model, os.path.join(SAVE, "diabetes_model.pkl"))
joblib.dump(diabetes_scaler, os.path.join(SAVE, "diabetes_scaler.pkl"))
print(f"  Best: {diabetes_results['best_model']} {diabetes_results['best_accuracy']}%")

print("\n== HEART/CARDIO ==")
df_heart = pd.read_csv(os.path.join(DATA, "cardio_train.csv"), sep=";")
df_heart["age_years"] = (df_heart["age"] / 365.25).round(1)
df_heart = df_heart[(df_heart["ap_hi"] >= 60) & (df_heart["ap_hi"] <= 250) & (df_heart["ap_lo"] >= 40) & (df_heart["ap_lo"] <= 200)]
df_heart = df_heart[(df_heart["height"] >= 100) & (df_heart["height"] <= 220) & (df_heart["weight"] >= 30) & (df_heart["weight"] <= 200)]
df_heart["bmi"] = (df_heart["weight"] / (df_heart["height"] / 100) ** 2).round(1)
heart_features = [
    "age_years",
    "gender",
    "height",
    "weight",
    "bmi",
    "ap_hi",
    "ap_lo",
    "cholesterol",
    "gluc",
    "smoke",
    "alco",
    "active",
]
x_heart = df_heart[heart_features].values.astype(float)
y_heart = df_heart["cardio"].values

heart_scaler = StandardScaler()
x_heart_scaled = heart_scaler.fit_transform(x_heart)
x_train_h, x_test_h, y_train_h, y_test_h = train_test_split(
    x_heart_scaled, y_heart, test_size=0.2, random_state=42, stratify=y_heart
)

heart_results, heart_models, best_heart_model = train_pool(
    x_train_h, x_test_h, y_train_h, y_test_h, heart_features
)
heart_results["dataset"] = {
    "total": int(len(df_heart)),
    "features": len(heart_features),
    "train": int(len(x_train_h)),
    "test": int(len(x_test_h)),
}
heart_results["feature_labels"] = HEART_FEATURE_LABELS

joblib.dump(best_heart_model, os.path.join(SAVE, "heart_model.pkl"))
joblib.dump(heart_scaler, os.path.join(SAVE, "heart_scaler.pkl"))
print(f"  Best: {heart_results['best_model']} {heart_results['best_accuracy']}%")

metrics = {
    "diabetes": diabetes_results,
    "heart": heart_results,
    "scaler_params": {
        "diabetes": {
            "mean_": [float(value) for value in diabetes_scaler.mean_],
            "scale_": [float(value) for value in diabetes_scaler.scale_],
        },
        "heart": {
            "mean_": [float(value) for value in heart_scaler.mean_],
            "scale_": [float(value) for value in heart_scaler.scale_],
        },
    },
    "lr_coefs": {
        "diabetes": {
            "intercept": float(diabetes_models["Logistic Regression"].intercept_[0]),
            "coef": [float(value) for value in diabetes_models["Logistic Regression"].coef_[0]],
        },
        "heart": {
            "intercept": float(heart_models["Logistic Regression"].intercept_[0]),
            "coef": [float(value) for value in heart_models["Logistic Regression"].coef_[0]],
        },
    },
}

with open(os.path.join(SAVE, "metrics.json"), "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=2)

print("\nDone. Run: py app.py")
