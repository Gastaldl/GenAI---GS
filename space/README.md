---
title: Previsão de Desmatamento via Satélite
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: 1.58.0
app_file: app.py
python_version: "3.12"
pinned: false
license: mit
---

# Previsão de Desmatamento via Dados de Satélite

Dashboard preditivo (camada de IA da Global Solution 2026 · Indústria Espacial · GAIE).

A partir de indicadores ambientais derivados de observação da Terra por satélite
(World Bank / Landsat · MODIS · Sentinel), o app:

- prevê a **variação da cobertura florestal** do próximo ciclo (regressão — RandomForest);
- classifica o **risco de desmatamento** (classificação — XGBoost);
- explica cada previsão com **SHAP** (waterfall);
- mostra a **evolução histórica** e a **projeção**.

Conectado ao **ODS 13 (Ação Climática)**.
