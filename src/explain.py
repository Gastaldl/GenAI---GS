"""
explain.py
==========
Interpretabilidade com SHAP (TreeExplainer) sobre os pipelines treinados.

Como os modelos vivem dentro de um Pipeline (pre-processador + estimador), as
features sao transformadas pelo pre-processador antes de calcular os valores SHAP,
preservando os nomes das colunas (inclusive as one-hot).

Funcoes principais:
  - explain_instance(pipe, X_row)  -> Explanation de 1 linha (usado no app: waterfall)
  - save_global_regression_shap(...) -> beeswarm + barra (usado no notebook/relatorio)
  - save_global_classification_shap(...) -> barra por classe
"""
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
    """Remove os prefixos 'num__'/'cat__' do ColumnTransformer para exibicao."""
    out = []
    for n in names:
        n = str(n)
        for prefix in ("num__", "cat__"):
            if n.startswith(prefix):
                n = n[len(prefix):]
        out.append(n)
    return out


def transform_to_df(pipe, X: pd.DataFrame) -> pd.DataFrame:
    """Aplica o pre-processador do pipeline e devolve um DataFrame com nomes de feature."""
    prep = pipe.named_steps["prep"]
    Xt = prep.transform(X)
    names = _clean_names(prep.get_feature_names_out())
    return pd.DataFrame(Xt, columns=names, index=X.index)


def explain_instance(pipe, X_row: pd.DataFrame):
    """
    Retorna a Explanation SHAP de UMA observacao (espaco de features original).
    Para regressao -> Explanation 1D pronta para shap.plots.waterfall.
    """
    model = pipe.named_steps["model"]
    Xt = transform_to_df(pipe, X_row)
    explainer = shap.TreeExplainer(model)
    sv = explainer(Xt, check_additivity=False)
    return sv[0]


def save_global_regression_shap(pipe, X_sample: pd.DataFrame, max_display: int = 15) -> dict:
    """Gera beeswarm e barra de importancia global para o regressor."""
    model = pipe.named_steps["model"]
    Xt = transform_to_df(pipe, X_sample)
    explainer = shap.TreeExplainer(model)
    sv = explainer(Xt, check_additivity=False)

    paths = {}
    plt.figure()
    shap.plots.beeswarm(sv, max_display=max_display, show=False)
    plt.title("SHAP - impacto das features (regressao)")
    plt.tight_layout()
    p = FIG_DIR / "shap_regression_beeswarm.png"
    plt.savefig(p, dpi=120, bbox_inches="tight")
    plt.close()
    paths["beeswarm"] = p

    plt.figure()
    shap.plots.bar(sv, max_display=max_display, show=False)
    plt.title("SHAP - importancia media (regressao)")
    plt.tight_layout()
    p = FIG_DIR / "shap_regression_bar.png"
    plt.savefig(p, dpi=120, bbox_inches="tight")
    plt.close()
    paths["bar"] = p
    return paths


def save_global_classification_shap(pipe, X_sample: pd.DataFrame, class_labels, max_display: int = 15) -> dict:
    """Gera barra de importancia global do classificador (agregada entre classes)."""
    model = pipe.named_steps["model"]
    Xt = transform_to_df(pipe, X_sample)
    explainer = shap.TreeExplainer(model)
    sv = explainer(Xt, check_additivity=False)

    paths = {}
    plt.figure()
    # sv pode ser 3D (amostras, features, classes) -> a barra agrega o impacto absoluto
    shap.plots.bar(sv.mean(axis=2) if sv.values.ndim == 3 else sv, max_display=max_display, show=False)
    plt.title("SHAP - importancia media (classificacao)")
    plt.tight_layout()
    p = FIG_DIR / "shap_classification_bar.png"
    plt.savefig(p, dpi=120, bbox_inches="tight")
    plt.close()
    paths["bar"] = p
    return paths


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

    print("Gerando SHAP global da regressao...")
    print(save_global_regression_shap(reg, sample))
    print("Gerando SHAP global da classificacao...")
    print(save_global_classification_shap(clf, sample, meta["class_labels"]))
    print("Exemplo de explicacao local (1 observacao):")
    expl = explain_instance(reg, df[feats].iloc[[0]])
    print("base_value=", float(expl.base_values), "| n_features=", len(expl.values))
