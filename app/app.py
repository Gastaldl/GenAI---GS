"""
app.py - Dashboard preditivo (Streamlit)
========================================
Camada preditiva tabular da solucao espacial: previsao da variacao da cobertura
florestal e da classe de risco de desmatamento, com explicacao SHAP.

Self-contained: depende apenas de models/ e data/processed/dataset.csv (nao precisa
da pasta src/). Funciona tanto rodando da raiz do repo quanto do Hugging Face Spaces.

Deploy local:  streamlit run app/app.py
"""
from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st

# --------------------------------------------------------------------------- #
# Localizacao de arquivos (funciona na raiz do repo e no HF Space)
# --------------------------------------------------------------------------- #
APP_DIR = Path(__file__).resolve().parent
CANDIDATES = [APP_DIR.parent, APP_DIR, Path.cwd()]


def _find(rel: str) -> Path:
    for base in CANDIDATES:
        p = base / rel
        if p.exists():
            return p
    return CANDIDATES[0] / rel


MODELS_DIR = _find("models")
DATA_PATH = _find("data/processed/dataset.csv")

RISK_COLORS = {
    "Desmatamento Alto": "#e63946",
    "Estavel": "#457b9d",
    "Recuperacao": "#2a9d8f",
}
SLIDER_FEATURES = [
    ("forest_area_pct", "Cobertura florestal (% do territorio)"),
    ("agri_land_pct", "Area agricola (%)"),
    ("urban_pop_pct", "Populacao urbana (%)"),
    ("co2_per_capita", "CO2 per capita (t)"),
    ("gdp_per_capita", "PIB per capita (US$)"),
]


# --------------------------------------------------------------------------- #
# SHAP (helpers self-contained)
# --------------------------------------------------------------------------- #
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
    prep = pipe.named_steps["prep"]
    Xt = prep.transform(X)
    names = _clean_names(prep.get_feature_names_out())
    return pd.DataFrame(Xt, columns=names, index=X.index)


def explain_instance(pipe, X_row: pd.DataFrame):
    model = pipe.named_steps["model"]
    Xt = transform_to_df(pipe, X_row)
    explainer = shap.TreeExplainer(model)
    sv = explainer(Xt, check_additivity=False)
    return sv[0]


# --------------------------------------------------------------------------- #
# Carga (cacheada)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def load_artifacts():
    reg = joblib.load(MODELS_DIR / "best_regressor.joblib")
    clf = joblib.load(MODELS_DIR / "best_classifier.joblib")
    meta = joblib.load(MODELS_DIR / "metadata.joblib")
    return reg, clf, meta


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="Previsao de Desmatamento via Satelite", layout="wide")

try:
    reg, clf, meta = load_artifacts()
    df = load_dataset()
except FileNotFoundError:
    st.error(
        "Modelos ou dataset nao encontrados. Rode antes:\n"
        "`python src/data_loader.py` -> `python src/preprocessing.py` -> `python src/train.py`"
    )
    st.stop()

numeric = meta["numeric_features"]
categorical = meta["categorical_features"]
feats = meta["feature_order"]
class_labels = meta["class_labels"]

# ---- Cabecalho / identidade ----
st.title("Previsao de Desmatamento via Dados de Satelite")
st.markdown(
    """
**Global Solution 2026 · Industria Espacial · Generative AI for Engineering**

Camada de **IA preditiva** de um produto de observacao da Terra: a partir de indicadores
ambientais derivados de **satelite** (World Bank / Landsat · MODIS · Sentinel), o sistema
projeta a **variacao da cobertura florestal** do proximo ciclo e classifica o **risco de
desmatamento** de cada pais/regiao. Conectado ao **ODS 13 (Acao Climatica)**.
"""
)

# ---- Sidebar: selecao ----
st.sidebar.header("Selecione a regiao")
countries = sorted(df["country"].dropna().unique().tolist())
default_idx = countries.index("Brazil") if "Brazil" in countries else 0
country = st.sidebar.selectbox("Pais/Regiao", countries, index=default_idx)

country_df = df[df["country"] == country].sort_values("year")
years = country_df["year"].tolist()
year = st.sidebar.selectbox("Ano base (estado atual)", years, index=len(years) - 1)

base_row = country_df[country_df["year"] == year].iloc[0]

st.sidebar.markdown("---")
st.sidebar.subheader("Ajuste de cenario (opcional)")
st.sidebar.caption("Altere os indicadores para simular cenarios alternativos.")

overrides = {}
for col, label in SLIDER_FEATURES:
    if col not in df.columns:
        continue
    lo = float(df[col].quantile(0.01))
    hi = float(df[col].quantile(0.99))
    cur = float(base_row[col])
    cur = min(max(cur, lo), hi)
    step = (hi - lo) / 100 if hi > lo else 1.0
    overrides[col] = st.sidebar.slider(label, lo, hi, cur, step=step)

# ---- Monta a linha de features (estado base + overrides) ----
X_row = base_row[feats].to_frame().T.copy()
for col, val in overrides.items():
    X_row[col] = val
for c in numeric:
    X_row[c] = pd.to_numeric(X_row[c], errors="coerce")

# ---- Predicoes ----
pred_pct = float(reg.predict(X_row[feats])[0])
clf_idx = int(clf.predict(X_row[feats])[0])
risk = class_labels[clf_idx]
proba = clf.predict_proba(X_row[feats])[0]

current_forest = float(base_row["forest_area_km2"])
projected_forest = current_forest * (1 + pred_pct / 100.0)

# ---- Saidas principais ----
c1, c2, c3 = st.columns(3)
c1.metric("Variacao prevista (proximo ciclo)", f"{pred_pct:+.2f}%",
          help="Variacao percentual da area de floresta prevista pelo regressor.")
c2.metric("Area de floresta projetada", f"{projected_forest:,.0f} km²",
          delta=f"{projected_forest - current_forest:,.0f} km²")
c3.markdown(
    f"<div style='padding:14px;border-radius:10px;background:{RISK_COLORS.get(risk,'#888')};"
    f"color:white;text-align:center'><b>Classe de risco</b><br>"
    f"<span style='font-size:1.4rem'>{risk}</span></div>",
    unsafe_allow_html=True,
)

st.markdown("### Evolucao historica e projecao")
col_left, col_right = st.columns([3, 2])

with col_left:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=country_df["year"], y=country_df["forest_area_km2"],
        mode="lines+markers", name="Historico", line=dict(color="#2a9d8f"),
    ))
    fig.add_trace(go.Scatter(
        x=[year, year + 1], y=[current_forest, projected_forest],
        mode="lines+markers", name="Projecao", line=dict(color="#e76f51", dash="dash"),
    ))
    fig.update_layout(
        xaxis_title="Ano", yaxis_title="Area de floresta (km²)",
        height=380, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h"),
    )
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    proba_df = pd.DataFrame({"Classe": class_labels, "Probabilidade": proba}).set_index("Classe")
    st.markdown("**Probabilidade por classe de risco**")
    st.bar_chart(proba_df)

# ---- Explicacao SHAP (por que o modelo decidiu isso) ----
st.markdown("### Por que o modelo previu isso? (SHAP)")
st.caption(
    "O grafico mostra como cada indicador empurrou a previsao de variacao florestal "
    "para cima (recuperacao) ou para baixo (desmatamento) nesta regiao."
)
with st.spinner("Calculando contribuicoes SHAP..."):
    expl = explain_instance(reg, X_row[feats])
    shap.plots.waterfall(expl, max_display=12, show=False)
    fig_shap = plt.gcf()
    fig_shap.set_size_inches(9, 5)
    st.pyplot(fig_shap, clear_figure=True)

# ---- Transparencia: metricas dos modelos ----
with st.expander("Detalhes tecnicos e desempenho dos modelos"):
    st.write(
        f"**Melhor regressor:** {meta['best_regressor']}  ·  "
        f"**Melhor classificador:** {meta['best_classifier']}  ·  "
        f"Treino ate {meta['train_cutoff']} (split temporal)."
    )
    st.markdown("**Comparacao - Regressao**")
    st.dataframe(pd.DataFrame(meta["regression_results"]), use_container_width=True)
    st.markdown("**Comparacao - Classificacao**")
    st.dataframe(pd.DataFrame(meta["classification_results"]), use_container_width=True)
    st.caption(
        "Fonte: World Bank Indicators API (indicadores ambientais derivados de observacao "
        "da Terra por satelite). Alvo: variacao % da area de floresta no ano seguinte."
    )

st.markdown("---")
st.caption("FIAP Global Solution 2026 · GAIE · ODS 13")
