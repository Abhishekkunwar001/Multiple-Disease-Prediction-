"""
Interview-ready heart disease training pipeline.

This script is focused on the cardio_train.csv dataset included in the project.
It applies:
- data analysis
- outlier treatment
- feature engineering
- feature selection
- class-balance comparison (class weights vs SMOTE)
- hyperparameter tuning
- cross validation
- soft-voting ensemble
- artifact and visualization export

Outputs are written into ../saved_models/heart_optimized/
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


BASE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE, "..", "datasets", "cardio_train.csv")
SAVE_DIR = os.path.join(BASE, "..", "saved_models", "heart_optimized")
os.makedirs(SAVE_DIR, exist_ok=True)
CV_SPLITS = 3
SEARCH_ITERATIONS = 6


@dataclass
class RunResult:
    name: str
    estimator: object
    cv_accuracy: float
    test_accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    y_pred: np.ndarray
    y_proba: np.ndarray


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, sep=";")
    return df


def analyze_dataset(df: pd.DataFrame) -> dict:
    return {
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "missing_values": df.isna().sum().to_dict(),
        "class_distribution": df["cardio"].value_counts().sort_index().to_dict(),
        "numeric_summary": df.describe().round(3).to_dict(),
        "correlation_with_target": df.corr(numeric_only=True)["cardio"].sort_values(ascending=False).round(4).to_dict(),
    }


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    clean = df.copy()
    clean["age_years"] = clean["age"] / 365.25

    numeric_filters = {
        "ap_hi": (60, 250),
        "ap_lo": (40, 200),
        "height": (100, 220),
        "weight": (30, 200),
    }
    for column, (low, high) in numeric_filters.items():
        clean = clean[(clean[column] >= low) & (clean[column] <= high)]

    return clean.reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    feat = df.copy()
    feat["bmi"] = feat["weight"] / ((feat["height"] / 100) ** 2)
    feat["pulse_pressure"] = feat["ap_hi"] - feat["ap_lo"]
    feat["mean_arterial_pressure"] = (feat["ap_hi"] + 2 * feat["ap_lo"]) / 3
    feat["bp_ratio"] = feat["ap_hi"] / np.clip(feat["ap_lo"], 1, None)
    feat["chol_gluc_risk"] = feat["cholesterol"] * feat["gluc"]
    feat["age_bp_ratio"] = feat["age_years"] / np.clip(feat["ap_hi"], 1, None)
    feat["age_chol_ratio"] = feat["age_years"] / np.clip(feat["cholesterol"], 1, None)
    feat["weight_height_ratio"] = feat["weight"] / np.clip(feat["height"], 1, None)
    feat["activity_smoke_interaction"] = (1 - feat["active"]) * feat["smoke"]
    feat["age_group"] = pd.cut(
        feat["age_years"],
        bins=[0, 40, 55, 120],
        labels=False,
        include_lowest=True,
    ).fillna(1).astype(int)
    feat["bmi_group"] = pd.cut(
        feat["bmi"],
        bins=[0, 18.5, 25, 30, 100],
        labels=False,
        include_lowest=True,
    ).fillna(1).astype(int)

    # The current dataset does not include heart-rate features, so we use
    # pressure-based alternatives instead of heart-rate ratios.
    return feat


def correlation_filter(df: pd.DataFrame, threshold: float = 0.95) -> list[str]:
    corr = df.corr(numeric_only=True).abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop_cols = [column for column in upper.columns if any(upper[column] > threshold)]
    return drop_cols


def build_preprocessor(feature_names: list[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer([("num", numeric_pipeline, feature_names)], remainder="drop")


def evaluate_pipeline(name: str, pipeline, x_train, y_train, x_test, y_test, cv) -> RunResult:
    cv_scores = cross_val_score(pipeline, x_train, y_train, cv=cv, scoring="accuracy", n_jobs=1)
    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)
    y_proba = pipeline.predict_proba(x_test)[:, 1]

    return RunResult(
        name=name,
        estimator=pipeline,
        cv_accuracy=float(np.mean(cv_scores)),
        test_accuracy=float(accuracy_score(y_test, y_pred)),
        precision=float(precision_score(y_test, y_pred)),
        recall=float(recall_score(y_test, y_pred)),
        f1=float(f1_score(y_test, y_pred)),
        roc_auc=float(roc_auc_score(y_test, y_proba)),
        y_pred=y_pred,
        y_proba=y_proba,
    )


def make_pipeline(model, feature_names: list[str], use_smote: bool = False):
    preprocessor = build_preprocessor(feature_names)
    steps = [
        ("preprocessor", preprocessor),
        ("select", SelectKBest(score_func=mutual_info_classif, k=min(12, len(feature_names)))),
    ]
    if use_smote:
        steps.append(("smote", SMOTE(random_state=42)))
        steps.append(("model", model))
        return ImbPipeline(steps)

    steps.append(("model", model))
    return Pipeline(steps)


def tune_model(base_pipeline, param_distributions, x_train, y_train, cv):
    search = RandomizedSearchCV(
        estimator=base_pipeline,
        param_distributions=param_distributions,
        n_iter=SEARCH_ITERATIONS,
        scoring="accuracy",
        cv=cv,
        verbose=0,
        random_state=42,
        n_jobs=1,
    )
    search.fit(x_train, y_train)
    return search


def selected_feature_names(pipeline, feature_names: list[str]) -> list[str]:
    support = pipeline.named_steps["select"].get_support()
    return [name for name, keep in zip(feature_names, support) if keep]


def get_feature_importance(result: RunResult, feature_names: list[str]) -> dict:
    selected = selected_feature_names(result.estimator, feature_names)
    model = result.estimator.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = np.abs(model.coef_[0])
    else:
        return {}
    return {feature: float(value) for feature, value in zip(selected, values)}


def plot_confusion_matrix(result: RunResult, y_test, path: str):
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, result.y_pred, cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(f"{result.name} Confusion Matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_roc_curve(result: RunResult, y_test, path: str):
    fpr, tpr, _ = roc_curve(y_test, result.y_proba)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fpr, tpr, label=f"{result.name} AUC={result.roc_auc:.3f}", color="#0ea5e9", linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#94a3b8")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{result.name} ROC Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_feature_importance(importance: dict, path: str, title: str):
    if not importance:
        return
    items = sorted(importance.items(), key=lambda item: item[1], reverse=True)[:12]
    labels = [item[0] for item in items]
    values = [item[1] for item in items]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(labels[::-1], values[::-1], color="#22c55e")
    ax.set_title(title)
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_comparison(results: list[RunResult], path: str):
    table = pd.DataFrame(
        [
            {
                "model": result.name,
                "cv_accuracy": round(result.cv_accuracy * 100, 2),
                "test_accuracy": round(result.test_accuracy * 100, 2),
                "precision": round(result.precision * 100, 2),
                "recall": round(result.recall * 100, 2),
                "f1": round(result.f1 * 100, 2),
                "roc_auc": round(result.roc_auc * 100, 2),
            }
            for result in results
        ]
    ).sort_values("test_accuracy", ascending=False)
    table.to_csv(path, index=False)
    return table
def main():
    df_raw = load_dataset()
    analysis = analyze_dataset(df_raw)

    df = clean_dataset(df_raw)
    df = engineer_features(df)

    y = df["cardio"].copy()
    x = df.drop(columns=["cardio", "id", "age"])

    to_drop = correlation_filter(x, threshold=0.95)
    x = x.drop(columns=to_drop)
    feature_names = list(x.columns)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=42)

    baseline_lr = evaluate_pipeline(
        "Logistic Regression (balanced)",
        make_pipeline(
            LogisticRegression(max_iter=2500, class_weight="balanced", random_state=42),
            feature_names,
            use_smote=False,
        ),
        x_train,
        y_train,
        x_test,
        y_test,
        cv,
    )

    rf_search = tune_model(
        make_pipeline(
            RandomForestClassifier(class_weight="balanced_subsample", random_state=42, n_jobs=-1),
            feature_names,
            use_smote=False,
        ),
        {
            "model__n_estimators": [200, 350, 500],
            "model__max_depth": [8, 12, None],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2],
            "model__max_features": ["sqrt", 0.7],
        },
        x_train,
        y_train,
        cv,
    )

    tuned_rf = evaluate_pipeline(
        "Tuned Random Forest",
        rf_search.best_estimator_,
        x_train,
        y_train,
        x_test,
        y_test,
        cv,
    )

    xgb_search = tune_model(
        make_pipeline(
            XGBClassifier(
                random_state=42,
                eval_metric="logloss",
                tree_method="hist",
            ),
            feature_names,
            use_smote=False,
        ),
        {
            "model__n_estimators": [200, 300, 450],
            "model__max_depth": [3, 4, 5],
            "model__learning_rate": [0.03, 0.05, 0.08, 0.1],
            "model__subsample": [0.75, 0.9],
            "model__colsample_bytree": [0.75, 0.9],
            "model__min_child_weight": [1, 3, 5],
        },
        x_train,
        y_train,
        cv,
    )

    tuned_xgb = evaluate_pipeline(
        "Tuned XGBoost",
        xgb_search.best_estimator_,
        x_train,
        y_train,
        x_test,
        y_test,
        cv,
    )

    smote_rf = evaluate_pipeline(
        "Random Forest + SMOTE",
        make_pipeline(
            RandomForestClassifier(
                n_estimators=400,
                max_depth=12,
                min_samples_leaf=2,
                class_weight=None,
                random_state=42,
                n_jobs=-1,
            ),
            feature_names,
            use_smote=True,
        ),
        x_train,
        y_train,
        x_test,
        y_test,
        cv,
    )

    voting_model = VotingClassifier(
        estimators=[
            ("lr", LogisticRegression(max_iter=2500, class_weight="balanced", random_state=42)),
            ("rf", rf_search.best_estimator_.named_steps["model"]),
            ("xgb", xgb_search.best_estimator_.named_steps["model"]),
        ],
        voting="soft",
    )

    voting_result = evaluate_pipeline(
        "Soft Voting Ensemble",
        Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(feature_names)),
                ("select", SelectKBest(score_func=mutual_info_classif, k=min(12, len(feature_names)))),
                ("model", voting_model),
            ]
        ),
        x_train,
        y_train,
        x_test,
        y_test,
        cv,
    )

    results = [baseline_lr, tuned_rf, tuned_xgb, smote_rf, voting_result]
    comparison = save_comparison(results, os.path.join(SAVE_DIR, "model_comparison.csv"))
    best_result = max(results, key=lambda result: result.test_accuracy)

    feature_importance = get_feature_importance(best_result, feature_names)
    plot_confusion_matrix(best_result, y_test, os.path.join(SAVE_DIR, "best_confusion_matrix.png"))
    plot_roc_curve(best_result, y_test, os.path.join(SAVE_DIR, "best_roc_curve.png"))
    plot_feature_importance(
        feature_importance,
        os.path.join(SAVE_DIR, "best_feature_importance.png"),
        f"{best_result.name} Feature Importance",
    )

    joblib.dump(best_result.estimator, os.path.join(SAVE_DIR, "heart_optimized_pipeline.joblib"))

    summary = {
        "analysis": analysis,
        "dropped_high_correlation_features": to_drop,
        "rf_best_params": rf_search.best_params_,
        "xgb_best_params": xgb_search.best_params_,
        "best_model": best_result.name,
        "best_metrics": {
            "cv_accuracy": round(best_result.cv_accuracy * 100, 2),
            "test_accuracy": round(best_result.test_accuracy * 100, 2),
            "precision": round(best_result.precision * 100, 2),
            "recall": round(best_result.recall * 100, 2),
            "f1": round(best_result.f1 * 100, 2),
            "roc_auc": round(best_result.roc_auc * 100, 2),
        },
        "comparison_table": comparison.to_dict(orient="records"),
        "feature_engineering_notes": [
            "age_years: interpretable age scale",
            "bmi: body-composition risk indicator",
            "pulse_pressure and mean_arterial_pressure: pressure dynamics",
            "bp_ratio: systolic-diastolic relationship",
            "chol_gluc_risk: combined metabolic burden",
            "age_bp_ratio and age_chol_ratio: age-normalized risk measures",
            "activity_smoke_interaction: joint lifestyle signal",
            "age_group and bmi_group: coarse non-linear risk bands",
            "Heart-rate ratio was not created because cardio_train.csv does not contain a heart-rate feature.",
        ],
    }

    with open(os.path.join(SAVE_DIR, "heart_optimized_summary.json"), "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("Saved optimized artifacts to:", SAVE_DIR)
    print("Best model:", best_result.name)
    print("Best test accuracy:", round(best_result.test_accuracy * 100, 2), "%")


if __name__ == "__main__":
    main()
