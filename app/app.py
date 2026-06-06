"""Dashboard Streamlit: previsao da variacao florestal, risco de desmatamento e SHAP.

Self-contained (so precisa de models/ e data/processed/dataset.csv). Local: streamlit run app/app.py
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

APP_DIR = Path(__file__).resolve().parent
CANDIDATES = [APP_DIR.parent, APP_DIR, Path.cwd()]  # raiz do repo ou raiz do HF Space


def _find(rel: str) -> Path:
    for base in CANDIDATES:
        if (base / rel).exists():
            return base / rel
    return CANDIDATES[0] / rel


MODELS_DIR = _find("models")
DATA_PATH = _find("data/processed/dataset.csv")

RISK_COLORS = {"Desmatamento Alto": "#e63946", "Estável": "#457b9d", "Recuperação": "#2a9d8f"}
SLIDER_FEATURES = [
    ("forest_area_pct", "Cobertura florestal (% do território)"),
    ("agri_land_pct", "Área agrícola (%)"),
    ("urban_pop_pct", "População urbana (%)"),
    ("co2_per_capita", "CO₂ per capita (t)"),
    ("gdp_per_capita", "PIB per capita (US$)"),
]


def _clean_names(names) -> list[str]:
    out = []
    for n in names:
        n = str(n)
        for prefix in ("num__", "cat__"):
            if n.startswith(prefix):
                n = n[len(prefix):]
        out.append(n)
    return out


def explain_instance(pipe, X_row: pd.DataFrame):
    prep = pipe.named_steps["prep"]
    Xt = pd.DataFrame(prep.transform(X_row), columns=_clean_names(prep.get_feature_names_out()), index=X_row.index)
    return shap.TreeExplainer(pipe.named_steps["model"])(Xt, check_additivity=False)[0]


@st.cache_resource(show_spinner=False)
def load_artifacts():
    return (
        joblib.load(MODELS_DIR / "best_regressor.joblib"),
        joblib.load(MODELS_DIR / "best_classifier.joblib"),
        joblib.load(MODELS_DIR / "metadata.joblib"),
    )


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


st.set_page_config(page_title="Previsão de Desmatamento via Satélite", layout="wide")
# barra de rolagem sempre visivel: evita o "tremor" (loop de redimensionamento) dos graficos
st.markdown("<style>html { overflow-y: scroll; }</style>", unsafe_allow_html=True)

try:
    reg, clf, meta = load_artifacts()
    df = load_dataset()
except FileNotFoundError:
    st.error("Modelos/dataset nao encontrados. Rode src/data_loader.py -> preprocessing.py -> train.py.")
    st.stop()

numeric = meta["numeric_features"]
feats = meta["feature_order"]
class_labels = meta["class_labels"]

st.title("Previsão de Desmatamento via Dados de Satélite")
st.markdown(
    """
**Global Solution 2026 · Indústria Espacial · Generative AI for Engineering**

Camada de **IA preditiva** de um produto de observação da Terra: a partir de indicadores
ambientais derivados de **satélite** (World Bank / Landsat · MODIS · Sentinel), o sistema
projeta a **variação da cobertura florestal** do próximo ciclo e classifica o **risco de
desmatamento** de cada país/região. Conectado ao **ODS 13 (Ação Climática)**.
"""
)

st.sidebar.header("Selecione a região")
countries = sorted(df["country"].dropna().unique().tolist())
default_idx = countries.index("Brazil") if "Brazil" in countries else 0
country = st.sidebar.selectbox("País/Região", countries, index=default_idx)

country_df = df[df["country"] == country].sort_values("year")
years = country_df["year"].tolist()
year = st.sidebar.selectbox("Ano base (estado atual)", years, index=len(years) - 1)
base_row = country_df[country_df["year"] == year].iloc[0]

st.sidebar.markdown("---")
st.sidebar.subheader("Ajuste de cenário (opcional)")
st.sidebar.caption("Altere os indicadores para simular cenários alternativos.")

overrides = {}
for col, label in SLIDER_FEATURES:
    if col not in df.columns:
        continue
    lo, hi = float(df[col].quantile(0.01)), float(df[col].quantile(0.99))
    cur = min(max(float(base_row[col]), lo), hi)
    step = (hi - lo) / 100 if hi > lo else 1.0
    overrides[col] = st.sidebar.slider(label, lo, hi, cur, step=step)

X_row = base_row[feats].to_frame().T.copy()
for col, val in overrides.items():
    X_row[col] = val
for c in numeric:
    X_row[c] = pd.to_numeric(X_row[c], errors="coerce")

pred_pct = float(reg.predict(X_row[feats])[0])
risk = class_labels[int(clf.predict(X_row[feats])[0])]
proba = clf.predict_proba(X_row[feats])[0]
current_forest = float(base_row["forest_area_km2"])
projected_forest = current_forest * (1 + pred_pct / 100.0)

c1, c2, c3 = st.columns(3)
c1.metric("Variação prevista (próximo ciclo)", f"{pred_pct:+.2f}%",
          help="Variação percentual da área de floresta prevista pelo regressor.")
c2.metric("Área de floresta projetada", f"{projected_forest:,.0f} km²",
          delta=f"{projected_forest - current_forest:,.0f} km²")
c3.markdown(
    f"<div style='padding:14px;border-radius:10px;background:{RISK_COLORS.get(risk,'#888')};"
    f"color:white;text-align:center'><b>Classe de risco</b><br>"
    f"<span style='font-size:1.4rem'>{risk}</span></div>",
    unsafe_allow_html=True,
)

st.markdown("### Evolução histórica e projeção")
col_left, col_right = st.columns([3, 2])
with col_left:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=country_df["year"], y=country_df["forest_area_km2"],
                             mode="lines+markers", name="Histórico", line=dict(color="#2a9d8f")))
    fig.add_trace(go.Scatter(x=[year, year + 1], y=[current_forest, projected_forest],
                             mode="lines+markers", name="Projeção", line=dict(color="#e76f51", dash="dash")))
    fig.update_layout(xaxis_title="Ano", yaxis_title="Área de floresta (km²)",
                      height=380, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)
with col_right:
    st.markdown("**Probabilidade por classe de risco**")
    st.bar_chart(pd.DataFrame({"Classe": class_labels, "Probabilidade": proba}).set_index("Classe"))

st.markdown("### Por que o modelo previu isso? (SHAP)")
st.caption(
    "O gráfico mostra como cada indicador empurrou a previsão de variação florestal "
    "para cima (recuperação) ou para baixo (desmatamento) nesta região."
)
with st.spinner("Calculando contribuições SHAP..."):
    shap.plots.waterfall(explain_instance(reg, X_row[feats]), max_display=12, show=False)
    fig_shap = plt.gcf()
    fig_shap.set_size_inches(9, 5)
    st.pyplot(fig_shap, clear_figure=True)

with st.expander("Detalhes técnicos e desempenho dos modelos"):
    st.write(
        f"**Melhor regressor:** {meta['best_regressor']}  ·  "
        f"**Melhor classificador:** {meta['best_classifier']}  ·  "
        f"Treino até {meta['train_cutoff']} (split temporal)."
    )
    st.markdown("**Comparação - Regressão**")
    st.dataframe(pd.DataFrame(meta["regression_results"]), use_container_width=True)
    st.markdown("**Comparação - Classificação**")
    st.dataframe(pd.DataFrame(meta["classification_results"]), use_container_width=True)
    st.caption(
        "Fonte: World Bank Indicators API (indicadores derivados de observação da Terra por "
        "satélite). Alvo: variação % da área de floresta no ano seguinte."
    )

st.markdown("---")
st.caption("FIAP Global Solution 2026 · GAIE · ODS 13")
