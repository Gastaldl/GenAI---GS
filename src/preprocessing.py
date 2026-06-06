"""Limpeza, imputacao, engenharia de atributos e alvos (regressao + classificacao).

Sem vazamento temporal: features sao do ano t (ou anteriores, via lags); alvos olham para t+1.
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

RAW_INDICATORS = [
    "forest_area_km2", "forest_area_pct", "agri_land_pct", "arable_land_pct",
    "urban_pop_pct", "pop_density", "population", "gdp_per_capita",
    "co2_per_capita", "renewable_energy_pct", "pm25_pollution", "land_area_km2",
]

# subconjunto usado para gerar lags/deltas
DYNAMIC_INDICATORS = [
    "forest_area_pct", "agri_land_pct", "urban_pop_pct",
    "co2_per_capita", "gdp_per_capita", "pm25_pollution",
]

# limiares (% ao ano) das classes de risco e clip de outliers do alvo
DEFOREST_THRESHOLD = -0.30
RECOVERY_THRESHOLD = 0.30
TARGET_CLIP = 25.0

# colunas que nunca sao features
ID_COLS = ["iso3", "country"]
TARGET_COLS = ["target_reg", "target_clf"]
HELPER_COLS = ["forest_next_km2"]
CATEGORICAL_FEATURES = ["region", "income"]


def load_panel() -> pd.DataFrame:
    if RAW_PATH.exists():
        df = pd.read_csv(RAW_PATH)
        logger.info("Painel bruto carregado de %s (%d linhas).", RAW_PATH, len(df))
    else:
        df = load_data()
    return df.sort_values(["iso3", "year"]).reset_index(drop=True)


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Interpolacao temporal por pais + mediana global para o que sobrar."""
    df = df.copy()
    num_cols = RAW_INDICATORS
    df[num_cols] = df.groupby("iso3")[num_cols].transform(
        lambda s: s.interpolate(method="linear", limit_direction="both")
    )
    df[num_cols] = df[num_cols].fillna(df[num_cols].median(numeric_only=True))
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Lags, variacoes ano-a-ano, media movel e razoes de pressao."""
    df = df.copy()
    g = df.groupby("iso3")
    for col in DYNAMIC_INDICATORS:
        df[f"{col}_lag1"] = g[col].shift(1)
        df[f"{col}_yoy"] = g[col].diff(1)
    df["forest_pct_roll3"] = g["forest_area_pct"].transform(
        lambda s: s.rolling(3, min_periods=1).mean()
    )
    eps = 1e-6
    df["agri_forest_ratio"] = df["agri_land_pct"] / (df["forest_area_pct"] + eps)
    df["urban_pressure"] = df["urban_pop_pct"] * np.log1p(df["pop_density"])
    df["dev_index"] = np.log1p(df["gdp_per_capita"]) * df["renewable_energy_pct"] / 100.0

    lag_cols = [f"{c}_lag1" for c in DYNAMIC_INDICATORS]
    df[lag_cols] = df.groupby("iso3")[lag_cols].transform(lambda s: s.bfill())
    df[lag_cols] = df[lag_cols].fillna(df[lag_cols].median())
    df[[f"{c}_yoy" for c in DYNAMIC_INDICATORS]] = df[[f"{c}_yoy" for c in DYNAMIC_INDICATORS]].fillna(0.0)
    return df


def build_targets(df: pd.DataFrame) -> pd.DataFrame:
    """target_reg = variacao % da floresta em t+1; target_clf = classe de risco."""
    df = df.copy()
    df["forest_next_km2"] = df.groupby("iso3")["forest_area_km2"].shift(-1)
    denom = df["forest_area_km2"].replace(0, np.nan)
    df["target_reg"] = ((df["forest_next_km2"] - df["forest_area_km2"]) / denom * 100.0).clip(-TARGET_CLIP, TARGET_CLIP)

    def classify(pct: float):
        if pd.isna(pct):
            return np.nan
        if pct < DEFOREST_THRESHOLD:
            return "Desmatamento Alto"
        if pct > RECOVERY_THRESHOLD:
            return "Recuperação"
        return "Estável"

    df["target_clf"] = df["target_reg"].apply(classify)
    return df


def get_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    exclude = set(ID_COLS + TARGET_COLS + HELPER_COLS + CATEGORICAL_FEATURES)
    numeric = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    return numeric, categorical


def preprocess(save: bool = True) -> pd.DataFrame:
    df = build_targets(engineer_features(impute_missing(load_panel())))
    if save:
        df.to_csv(PROCESSED_PATH, index=False)
        logger.info("Dataset processado salvo em %s", PROCESSED_PATH)
    return df


if __name__ == "__main__":
    data = preprocess()
    num, cat = get_feature_columns(data)
    modeling = data.dropna(subset=TARGET_COLS)
    print(f"Dataset processado : {data.shape[0]} linhas x {data.shape[1]} colunas")
    print(f"Linhas modelaveis  : {len(modeling)} | features: {len(num)} num + {len(cat)} cat")
    print("\nDistribuicao das classes (target_clf):")
    print(modeling["target_clf"].value_counts())
    print("\nAlvo de regressao (target_reg, % ao ano):")
    print(modeling["target_reg"].describe())
