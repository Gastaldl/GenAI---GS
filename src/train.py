"""Treina e compara modelos de regressao e classificacao; salva os melhores em models/.

Validacao: split temporal (teste em anos recentes) + GroupKFold por pais (sem vazamento).
"""
from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

import evaluate
from preprocessing import PROCESSED_PATH, TARGET_COLS, get_feature_columns, preprocess

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train")

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CUTOFF = 2016  # treino: year <= 2016 ; teste: year > 2016
CV_SPLITS = 5
RANDOM_STATE = 42

try:
    from xgboost import XGBClassifier, XGBRegressor

    HAS_XGB = True
except Exception:
    HAS_XGB = False
    logger.warning("XGBoost indisponivel; usando apenas scikit-learn.")


def build_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [("num", numeric_pipe, numeric), ("cat", categorical_pipe, categorical)], remainder="drop"
    )


def regression_models() -> dict:
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=200, max_depth=16, min_samples_leaf=4, n_jobs=-1, random_state=RANDOM_STATE
        ),
        "GradientBoosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBRegressor(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, random_state=RANDOM_STATE, n_jobs=-1,
        )
    return models


def classification_models() -> dict:
    models = {
        "LogisticRegression": LogisticRegression(max_iter=2000),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, n_jobs=-1, random_state=RANDOM_STATE, class_weight="balanced"
        ),
        "GradientBoosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.9,
            colsample_bytree=0.9, random_state=RANDOM_STATE, n_jobs=-1, eval_metric="mlogloss",
        )
    return models


def temporal_split(df: pd.DataFrame):
    return df[df["year"] <= TRAIN_CUTOFF], df[df["year"] > TRAIN_CUTOFF]


def run_regression(df, numeric, categorical):
    logger.info("=== Regressão ===")
    train, test = temporal_split(df)
    X_train, y_train = train[numeric + categorical], train["target_reg"]
    X_test, y_test = test[numeric + categorical], test["target_reg"]
    gkf = GroupKFold(n_splits=CV_SPLITS)

    rows, fitted = [], {}
    for name, model in regression_models().items():
        pipe = Pipeline([("prep", build_preprocessor(numeric, categorical)), ("model", model)])
        cv_r2 = cross_val_score(pipe, X_train, y_train, groups=train["iso3"], cv=gkf, scoring="r2")
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        rows.append({
            "model": name,
            "cv_r2_mean": float(np.mean(cv_r2)),
            "cv_r2_std": float(np.std(cv_r2)),
            "test_r2": float(r2_score(y_test, pred)),
            "test_mae": float(mean_absolute_error(y_test, pred)),
            "test_rmse": float(root_mean_squared_error(y_test, pred)),
        })
        fitted[name] = pipe
        logger.info("%-18s CV R2=%.3f | test R2=%.3f RMSE=%.3f",
                    name, np.mean(cv_r2), rows[-1]["test_r2"], rows[-1]["test_rmse"])

    results = pd.DataFrame(rows).sort_values("test_r2", ascending=False).reset_index(drop=True)
    best_name = results.iloc[0]["model"]
    evaluate.plot_regression_scatter(y_test, fitted[best_name].predict(X_test), "regression_scatter")
    evaluate.plot_model_comparison(results, "test_r2", "Regressão - R² no teste", "regression_comparison")
    evaluate.save_metrics_table(results, "regression_comparison")
    return results, best_name


def run_classification(df, numeric, categorical):
    logger.info("=== Classificação ===")
    encoder = LabelEncoder().fit(df["target_clf"])
    df = df.assign(_y=encoder.transform(df["target_clf"]))
    train, test = temporal_split(df)
    X_train, y_train = train[numeric + categorical], train["_y"]
    X_test, y_test = test[numeric + categorical], test["_y"]
    gkf = GroupKFold(n_splits=CV_SPLITS)

    rows, fitted = [], {}
    for name, model in classification_models().items():
        pipe = Pipeline([("prep", build_preprocessor(numeric, categorical)), ("model", model)])
        cv_f1 = cross_val_score(pipe, X_train, y_train, groups=train["iso3"], cv=gkf, scoring="f1_macro")
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test)
        try:
            auc = float(roc_auc_score(y_test, proba, multi_class="ovr", average="macro"))
        except Exception:
            auc = float("nan")
        rows.append({
            "model": name,
            "cv_f1_macro_mean": float(np.mean(cv_f1)),
            "cv_f1_macro_std": float(np.std(cv_f1)),
            "test_accuracy": float(accuracy_score(y_test, pred)),
            "test_f1_macro": float(f1_score(y_test, pred, average="macro")),
            "test_roc_auc_ovr": auc,
        })
        fitted[name] = pipe
        logger.info("%-18s CV F1=%.3f | test acc=%.3f F1=%.3f AUC=%.3f",
                    name, np.mean(cv_f1), rows[-1]["test_accuracy"], rows[-1]["test_f1_macro"], auc)

    results = pd.DataFrame(rows).sort_values("test_f1_macro", ascending=False).reset_index(drop=True)
    best_name = results.iloc[0]["model"]
    best_pred = fitted[best_name].predict(X_test)
    print("\nRelatório de classificação (melhor modelo, conjunto de teste):")
    print(classification_report(y_test, best_pred, target_names=list(encoder.classes_)))
    evaluate.plot_confusion(
        encoder.inverse_transform(y_test), encoder.inverse_transform(best_pred),
        list(encoder.classes_), "confusion_matrix",
    )
    evaluate.plot_model_comparison(results, "test_f1_macro", "Classificação - F1 macro no teste", "classification_comparison")
    evaluate.save_metrics_table(results, "classification_comparison")
    return results, best_name, encoder


def main():
    df = pd.read_csv(PROCESSED_PATH) if PROCESSED_PATH.exists() else preprocess()
    df = df.dropna(subset=TARGET_COLS).reset_index(drop=True)
    numeric, categorical = get_feature_columns(df)
    logger.info("Linhas modeláveis: %d | features: %d num + %d cat", len(df), len(numeric), len(categorical))

    reg_results, best_reg = run_regression(df, numeric, categorical)
    clf_results, best_clf, encoder = run_classification(df, numeric, categorical)

    # refit dos melhores em todos os dados (mais dados -> melhor para o app)
    logger.info("Refit dos melhores modelos em todos os dados modeláveis...")
    X_all = df[numeric + categorical]
    best_reg_pipe = Pipeline(
        [("prep", build_preprocessor(numeric, categorical)), ("model", regression_models()[best_reg])]
    ).fit(X_all, df["target_reg"])
    best_clf_pipe = Pipeline(
        [("prep", build_preprocessor(numeric, categorical)), ("model", classification_models()[best_clf])]
    ).fit(X_all, encoder.transform(df["target_clf"]))

    joblib.dump(best_reg_pipe, MODELS_DIR / "best_regressor.joblib", compress=3)
    joblib.dump(best_clf_pipe, MODELS_DIR / "best_classifier.joblib", compress=3)
    metadata = {
        "numeric_features": numeric,
        "categorical_features": categorical,
        "feature_order": numeric + categorical,
        "class_labels": list(encoder.classes_),
        "best_regressor": best_reg,
        "best_classifier": best_clf,
        "train_cutoff": TRAIN_CUTOFF,
        "regression_results": reg_results.to_dict(orient="records"),
        "classification_results": clf_results.to_dict(orient="records"),
        "has_xgboost": HAS_XGB,
    }
    joblib.dump(metadata, MODELS_DIR / "metadata.joblib", compress=3)
    logger.info("Modelos salvos em %s", MODELS_DIR)

    print("\n===== RESUMO =====")
    print("Melhor regressor   :", best_reg)
    print(reg_results.to_string(index=False))
    print("\nMelhor classificador:", best_clf)
    print(clf_results.to_string(index=False))


if __name__ == "__main__":
    main()
