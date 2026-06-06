"""Baixa indicadores ambientais reais do World Bank e monta o painel country x year."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import requests
except ImportError:
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

# nome -> codigos candidatos (tenta na ordem; o CO2 mudou de codigo em 2024)
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


def _wb_get(path: str, params: dict | None = None) -> list:
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
    rows = _wb_get("country", {"per_page": 400})
    recs = [
        {
            "iso3": d.get("id"),
            "country": d.get("name"),
            "region": (d.get("region") or {}).get("value"),
            "income": (d.get("incomeLevel") or {}).get("value"),
        }
        for d in rows
    ]
    return pd.DataFrame(recs)


def fetch_indicator(codes: list[str]) -> tuple[pd.DataFrame, str]:
    for code in codes:
        rows = _wb_get(f"country/all/indicator/{code}", {"date": f"{START_YEAR}:{END_YEAR}"})
        recs = [
            {"iso3": d.get("countryiso3code"), "year": int(d["date"]), "value": d.get("value")}
            for d in rows
            if d.get("value") is not None and d.get("countryiso3code")
        ]
        if recs:
            return pd.DataFrame(recs), code
    return pd.DataFrame(columns=["iso3", "year", "value"]), codes[0]


def fetch_worldbank() -> pd.DataFrame:
    meta = fetch_country_metadata()
    if meta.empty:
        raise RuntimeError("Metadados do World Bank vieram vazios.")
    # remove agregados regionais (World, High income, etc.)
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
            df = df.merge(ind_df.rename(columns={"value": name}), on=["iso3", "year"], how="left")
    logger.info("Codigos do World Bank utilizados: %s", used_codes)
    return df


def load_data(save: bool = True) -> pd.DataFrame:
    """Painel real do World Bank; levanta erro claro se a API estiver indisponivel."""
    logger.info("Buscando dados reais no World Bank API...")
    try:
        df = fetch_worldbank()
    except Exception as exc:
        raise RuntimeError(
            f"Falha ao obter dados do World Bank API ({exc}). Verifique a internet e tente de novo."
        ) from exc

    if len(df) < 1000 or df["forest_area_km2"].notna().sum() < 500:
        raise RuntimeError(
            f"World Bank retornou dados insuficientes (linhas={len(df)}). Tente novamente mais tarde."
        )

    logger.info("World Bank OK: %d linhas.", len(df))
    if save:
        df.to_csv(RAW_PATH, index=False)
        logger.info("Painel salvo em %s", RAW_PATH)
    return df


def _validate(df: pd.DataFrame) -> None:
    feature_like = [c for c in df.columns if c not in META_COLS + ["year"]]
    assert len(df) >= 1000, f"Esperado >=1000 linhas, obtido {len(df)}"
    assert len(feature_like) >= 10, f"Esperado >=10 colunas, obtido {len(feature_like)}"


if __name__ == "__main__":
    panel = load_data()
    feats = [c for c in panel.columns if c not in META_COLS + ["year"]]
    print(f"Fonte    : World Bank Indicators API (real)")
    print(f"Shape    : {panel.shape[0]} linhas x {panel.shape[1]} colunas")
    print(f"Periodo  : {panel['year'].min()}-{panel['year'].max()} | paises: {panel['iso3'].nunique()}")
    _validate(panel)
    print(f"OK: {panel.shape[0]} linhas (>=1000) e {len(feats)} indicadores (>=10).")
