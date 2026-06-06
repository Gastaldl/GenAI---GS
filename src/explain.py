"""Interpretabilidade com SHAP (TreeExplainer) sobre os pipelines treinados."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import shap

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _clean_names(names) -> list[str]:
    out = []
    for n in names:
        n = str(n)
        for prefix in ("num__", "cat__"):
            if n.startswith(prefix):
                n = n[len(prefix):]
        out.append(n)
    return out


def transform_to_df(pipe, X: pd.DataFrame) -> pd.DataFrame:
    """Aplica o pre-processador e devolve um DataFrame com nomes de feature."""
    prep = pipe.named_steps["prep"]
    Xt = prep.transform(X)
    return pd.DataFrame(Xt, columns=_clean_names(prep.get_feature_names_out()), index=X.index)


def explain_instance(pipe, X_row: pd.DataFrame):
    """Explanation SHAP de 1 observacao (pronta para shap.plots.waterfall)."""
    Xt = transform_to_df(pipe, X_row)
    sv = shap.TreeExplainer(pipe.named_steps["model"])(Xt, check_additivity=False)
    return sv[0]


def save_global_regression_shap(pipe, X_sample: pd.DataFrame, max_display: int = 15) -> dict:
    Xt = transform_to_df(pipe, X_sample)
    sv = shap.TreeExplainer(pipe.named_steps["model"])(Xt, check_additivity=False)

    paths = {}
    plt.figure()
    shap.plots.beeswarm(sv, max_display=max_display, show=False)
    plt.title("SHAP - impacto das features (regressão)")
    plt.tight_layout()
    paths["beeswarm"] = FIG_DIR / "shap_regression_beeswarm.png"
    plt.savefig(paths["beeswarm"], dpi=120, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.plots.bar(sv, max_display=max_display, show=False)
    plt.title("SHAP - importância média (regressão)")
    plt.tight_layout()
    paths["bar"] = FIG_DIR / "shap_regression_bar.png"
    plt.savefig(paths["bar"], dpi=120, bbox_inches="tight")
    plt.close()
    return paths


def save_global_classification_shap(pipe, X_sample: pd.DataFrame, class_labels, max_display: int = 15) -> dict:
    Xt = transform_to_df(pipe, X_sample)
    sv = shap.TreeExplainer(pipe.named_steps["model"])(Xt, check_additivity=False)
    plt.figure()
    # sv pode ser 3D (amostras, features, classes) -> agrega entre classes
    shap.plots.bar(sv.mean(axis=2) if sv.values.ndim == 3 else sv, max_display=max_display, show=False)
    plt.title("SHAP - importância média (classificação)")
    plt.tight_layout()
    path = FIG_DIR / "shap_classification_bar.png"
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    return {"bar": path}


if __name__ == "__main__":
    import joblib

    from preprocessing import PROCESSED_PATH, TARGET_COLS, get_feature_columns

    df = pd.read_csv(PROCESSED_PATH).dropna(subset=TARGET_COLS)
    numeric, categorical = get_feature_columns(df)
    feats = numeric + categorical
    sample = df[feats].sample(min(400, len(df)), random_state=42)

    reg = joblib.load(ROOT / "models" / "best_regressor.joblib")
    clf = joblib.load(ROOT / "models" / "best_classifier.joblib")
    meta = joblib.load(ROOT / "models" / "metadata.joblib")

    print(save_global_regression_shap(reg, sample))
    print(save_global_classification_shap(clf, sample, meta["class_labels"]))
    expl = explain_instance(reg, df[feats].iloc[[0]])
    print("base_value=", float(expl.base_values), "| n_features=", len(expl.values))
