"""
data_loader.py
==============
Carrega indicadores ambientais/socioeconomicos REAIS do World Bank (muitos deles
derivados de observacao da Terra por satelite - Landsat/MODIS/Sentinel) e monta um
painel `country x year`.

Saida: data/raw/worldbank_panel.csv  (painel largo, antes da engenharia de atributos)

Conexao com a Industria Espacial: os indicadores de cobertura florestal/uso do solo
do World Bank sao produzidos a partir de sensoriamento remoto por satelite. Este
modulo e a porta de entrada de um "produto de dados espaciais".

Observacao: usamos exclusivamente dados REAIS. Se a API estiver indisponivel, o modulo
levanta um erro claro (sem gerar dados falsos), preservando a integridade do projeto.
Os dados ja baixados ficam versionados em data/raw/, permitindo rodar o restante do
pipeline (preprocessing/treino/app) offline.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import requests
except ImportError:  # pragma: no cover - requests faz parte do requirements
    requests = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("data_loader")

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)
RAW_PATH = RAW_DIR / "worldbank_panel.csv"

WB_BASE = "https://api.worldbank.org/v2"
START_YEAR = 2000
END_YEAR = 2022

# Nome amigavel -> lista de codigos candidatos do World Bank (tenta na ordem).
# Alguns codigos mudaram em 2024 (ex.: CO2), por isso ha alternativas.
INDICATORS: list[tuple[str, list[str]]] = [
    ("forest_area_km2", ["AG.LND.FRST.K2"]),
    ("forest_area_pct", ["AG.LND.FRST.ZS"]),
    ("agri_land_pct", ["AG.LND.AGRI.ZS"]),
    ("arable_land_pct", ["AG.LND.ARBL.ZS"]),
    ("urban_pop_pct", ["SP.URB.TOTL.IN.ZS"]),
    ("pop_density", ["EN.POP.DNST"]),
    ("population", ["SP.POP.TOTL"]),
    ("gdp_per_capita", ["NY.GDP.PCAP.CD"]),
    ("co2_per_capita", ["EN.GHG.CO2.PC.CE.AR5", "EN.ATM.CO2E.PC"]),
    ("renewable_energy_pct", ["EG.FEC.RNEW.ZS"]),
    ("pm25_pollution", ["EN.ATM.PM25.MC.M3"]),
    ("land_area_km2", ["AG.LND.TOTL.K2"]),
]

META_COLS = ["iso3", "country", "region", "income"]
INDICATOR_NAMES = [name for name, _ in INDICATORS]


# --------------------------------------------------------------------------- #
# World Bank API (real)
# --------------------------------------------------------------------------- #
def _wb_get(path: str, params: dict | None = None) -> list:
    """Faz uma chamada GET ao World Bank e devolve a lista de registros (data[1])."""
    if requests is None:
        raise RuntimeError("Biblioteca 'requests' nao esta instalada.")
    params = {**(params or {}), "format": "json", "per_page": 20000}
    resp = requests.get(f"{WB_BASE}/{path}", params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, list) or len(data) < 2 or data[1] is None:
        return []
    return data[1]


def fetch_country_metadata() -> pd.DataFrame:
    """Lista de paises com regiao e faixa de renda; usada para remover agregados."""
    rows = _wb_get("country", {"per_page": 400})
    recs = []
    for d in rows:
        recs.append(
            {
                "iso3": d.get("id"),
                "country": d.get("name"),
                "region": (d.get("region") or {}).get("value"),
                "income": (d.get("incomeLevel") or {}).get("value"),
            }
        )
    return pd.DataFrame(recs)


def fetch_indicator(codes: list[str]) -> tuple[pd.DataFrame, str]:
    """Tenta cada codigo candidato e retorna o primeiro com dados."""
    for code in codes:
        rows = _wb_get(
            f"country/all/indicator/{code}", {"date": f"{START_YEAR}:{END_YEAR}"}
        )
        recs = [
            {
                "iso3": d.get("countryiso3code"),
                "year": int(d["date"]),
                "value": d.get("value"),
            }
            for d in rows
            if d.get("value") is not None and d.get("countryiso3code")
        ]
        if recs:
            return pd.DataFrame(recs), code
    return pd.DataFrame(columns=["iso3", "year", "value"]), codes[0]


def fetch_worldbank() -> pd.DataFrame:
    """Monta o painel largo (country x year) com todos os indicadores."""
    meta = fetch_country_metadata()
    if meta.empty:
        raise RuntimeError("Metadados do World Bank vieram vazios.")
    # remove agregados regionais (ex.: 'World', 'High income', etc.)
    meta = meta[meta["region"].notna() & (meta["region"] != "Aggregates")].copy()
    meta = meta.dropna(subset=["iso3"])

    base = meta[META_COLS].merge(
        pd.DataFrame({"year": range(START_YEAR, END_YEAR + 1)}), how="cross"
    )

    df = base
    used_codes: dict[str, str] = {}
    for name, codes in INDICATORS:
        ind_df, used = fetch_indicator(codes)
        used_codes[name] = used
        if ind_df.empty:
            logger.warning("Indicador '%s' (%s) sem dados.", name, codes)
            df[name] = np.nan
        else:
            df = df.merge(
                ind_df.rename(columns={"value": name}), on=["iso3", "year"], how="left"
            )
    logger.info("Codigos do World Bank utilizados: %s", used_codes)
    return df


# --------------------------------------------------------------------------- #
# Orquestracao
# --------------------------------------------------------------------------- #
def load_data(save: bool = True) -> pd.DataFrame:
    """Baixa o painel REAL do World Bank. Levanta erro claro se indisponivel (sem fallback)."""
    logger.info("Buscando dados reais no World Bank API...")
    try:
        df = fetch_worldbank()
    except Exception as exc:  # rede, timeout, mudanca de codigo, etc.
        raise RuntimeError(
            f"Falha ao obter dados do World Bank API ({exc}). "
            "Verifique a conexao com a internet e tente novamente. "
            "Dados ja baixados ficam em data/raw/worldbank_panel.csv."
        ) from exc

    ok_rows = len(df) >= 1000
    ok_target = df["forest_area_km2"].notna().sum() >= 500
    if not (ok_rows and ok_target):
        raise RuntimeError(
            f"World Bank retornou dados insuficientes (linhas={len(df)}, "
            f"forest_nao_nulo={int(df['forest_area_km2'].notna().sum())}). "
            "Tente novamente mais tarde."
        )

    logger.info("World Bank OK: %d linhas.", len(df))
    if save:
        df.to_csv(RAW_PATH, index=False)
        logger.info("Painel salvo em %s", RAW_PATH)
    return df


def _validate(df: pd.DataFrame) -> None:
    n_rows, _ = df.shape
    feature_like = [c for c in df.columns if c not in META_COLS + ["year"]]
    assert n_rows >= 1000, f"Esperado >=1000 linhas, obtido {n_rows}"
    assert len(feature_like) >= 10, f"Esperado >=10 colunas de dados, obtido {len(feature_like)}"


if __name__ == "__main__":
    panel = load_data()
    rows, cols = panel.shape
    feats = [c for c in panel.columns if c not in META_COLS + ["year"]]
    print(f"Fonte de dados : World Bank Indicators API (real)")
    print(f"Shape          : {rows} linhas x {cols} colunas")
    print(f"Indicadores    : {len(feats)} ({', '.join(feats)})")
    print(f"Periodo        : {panel['year'].min()}-{panel['year'].max()}")
    print(f"Paises/regioes : {panel['iso3'].nunique()}")
    print("\nPrevia:")
    print(panel.head())
    _validate(panel)
    print(f"\nOK: {rows} linhas (>=1000) e {len(feats)} indicadores (>=10).")
