"""
preprocessing.py
================
Limpeza, imputacao de valores ausentes, engenharia de atributos e criacao dos alvos.

Problema de ML (camada preditiva tabular da solucao espacial):
  - REGRESSAO  -> `target_reg`: variacao percentual da area de floresta no ANO SEGUINTE (%).
  - CLASSIFICACAO -> `target_clf`: classe de risco derivada da variacao:
        "Desmatamento Alto" | "Estavel" | "Recuperacao"

Sem vazamento temporal: todas as features sao do ano t (ou anteriores, via lags);
os alvos olham para t+1.

Saida: data/processed/dataset.csv  (painel + features + alvos)
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from data_loader import RAW_PATH, load_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("preprocessing")

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_PATH = PROCESSED_DIR / "dataset.csv"

# Indicadores brutos vindos do data_loader
RAW_INDICATORS = [
    "forest_area_km2",
    "forest_area_pct",
    "agri_land_pct",
    "arable_land_pct",
    "urban_pop_pct",
    "pop_density",
    "population",
    "gdp_per_capita",
    "co2_per_capita",
    "renewable_energy_pct",
    "pm25_pollution",
    "land_area_km2",
]

# Indicadores usados para gerar lags/deltas (subconjunto mais relevante)
DYNAMIC_INDICATORS = [
    "forest_area_pct",
    "agri_land_pct",
    "urban_pop_pct",
    "co2_per_capita",
    "gdp_per_capita",
    "pm25_pollution",
]

# Limiares (em % ao ano) para definir as classes de risco. Configuraveis.
DEFOREST_THRESHOLD = -0.30   # perda anual de floresta > 0.30% => risco alto
RECOVERY_THRESHOLD = 0.30    # ganho anual de floresta > 0.30%  => recuperacao
TARGET_CLIP = 25.0           # limita outliers extremos do alvo de regressao (%)

# Colunas que NUNCA sao features
ID_COLS = ["iso3", "country"]
TARGET_COLS = ["target_reg", "target_clf"]
HELPER_COLS = ["forest_next_km2"]
CATEGORICAL_FEATURES = ["region", "income"]


# --------------------------------------------------------------------------- #
def load_panel() -> pd.DataFrame:
    """Carrega o painel bruto; gera/baixa caso ainda nao exista."""
    if RAW_PATH.exists():
        df = pd.read_csv(RAW_PATH)
        logger.info("Painel bruto carregado de %s (%d linhas).", RAW_PATH, len(df))
    else:
        logger.info("Painel bruto inexistente; chamando load_data().")
        df = load_data()
    return df.sort_values(["iso3", "year"]).reset_index(drop=True)


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Imputa valores ausentes: interpolacao temporal por pais + mediana global."""
    df = df.copy()
    num_cols = RAW_INDICATORS
    # 1) interpolacao linear dentro de cada pais (preenche buracos no meio e pontas)
    df[num_cols] = df.groupby("iso3")[num_cols].transform(
        lambda s: s.interpolate(method="linear", limit_direction="both")
    )
    # 2) o que sobrar (pais sem nenhum valor para um indicador) -> mediana global
    medians = df[num_cols].median(numeric_only=True)
    df[num_cols] = df[num_cols].fillna(medians)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria lags, variacoes ano-a-ano, medias moveis e razoes."""
    df = df.copy()
    g = df.groupby("iso3")

    # Lags (estado do ano anterior)
    for col in DYNAMIC_INDICATORS:
        df[f"{col}_lag1"] = g[col].shift(1)

    # Variacao ano-a-ano (tendencia recente)
    for col in DYNAMIC_INDICATORS:
        df[f"{col}_yoy"] = g[col].diff(1)

    # Media movel de 3 anos da cobertura florestal
    df["forest_pct_roll3"] = g["forest_area_pct"].transform(
        lambda s: s.rolling(3, min_periods=1).mean()
    )

    # Razoes / indicadores de pressao
    eps = 1e-6
    df["agri_forest_ratio"] = df["agri_land_pct"] / (df["forest_area_pct"] + eps)
    df["urban_pressure"] = df["urban_pop_pct"] * np.log1p(df["pop_density"])
    df["dev_index"] = np.log1p(df["gdp_per_capita"]) * df["renewable_energy_pct"] / 100.0

    # Imputa NaNs gerados: lags -> backfill por pais + mediana; deltas -> 0 (sem variacao conhecida)
    lag_cols = [f"{c}_lag1" for c in DYNAMIC_INDICATORS]
    df[lag_cols] = df.groupby("iso3")[lag_cols].transform(lambda s: s.bfill())
    df[lag_cols] = df[lag_cols].fillna(df[lag_cols].median())
    yoy_cols = [f"{c}_yoy" for c in DYNAMIC_INDICATORS]
    df[yoy_cols] = df[yoy_cols].fillna(0.0)
    return df


def build_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Cria o alvo de regressao (% de variacao no ano seguinte) e o de classificacao."""
    df = df.copy()
    g = df.groupby("iso3")
    df["forest_next_km2"] = g["forest_area_km2"].shift(-1)
    denom = df["forest_area_km2"].replace(0, np.nan)
    df["target_reg"] = (df["forest_next_km2"] - df["forest_area_km2"]) / denom * 100.0
    df["target_reg"] = df["target_reg"].clip(-TARGET_CLIP, TARGET_CLIP)

    def classify(pct: float):
        if pd.isna(pct):
            return np.nan
        if pct < DEFOREST_THRESHOLD:
            return "Desmatamento Alto"
        if pct > RECOVERY_THRESHOLD:
            return "Recuperacao"
        return "Estavel"

    df["target_clf"] = df["target_reg"].apply(classify)
    return df


def get_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Retorna (features_numericas, features_categoricas) a partir do dataframe processado."""
    exclude = set(ID_COLS + TARGET_COLS + HELPER_COLS + CATEGORICAL_FEATURES)
    numeric = [
        c
        for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]
    categorical = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    return numeric, categorical


def preprocess(save: bool = True) -> pd.DataFrame:
    """Pipeline completo de preparacao dos dados."""
    df = load_panel()
    df = impute_missing(df)
    df = engineer_features(df)
    df = build_targets(df)

    if save:
        df.to_csv(PROCESSED_PATH, index=False)
        logger.info("Dataset processado salvo em %s", PROCESSED_PATH)
    return df


if __name__ == "__main__":
    data = preprocess()
    num, cat = get_feature_columns(data)
    modeling = data.dropna(subset=TARGET_COLS)
    print(f"Dataset processado : {data.shape[0]} linhas x {data.shape[1]} colunas")
    print(f"Linhas modelaveis  : {len(modeling)} (com alvo nao-nulo)")
    print(f"Features numericas : {len(num)}")
    print(f"Features categoricas: {cat}")
    print("\nDistribuicao das classes (target_clf):")
    print(modeling["target_clf"].value_counts())
    print("\nEstatisticas do alvo de regressao (target_reg, % ao ano):")
    print(modeling["target_reg"].describe())
