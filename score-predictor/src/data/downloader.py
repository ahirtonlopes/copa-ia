"""
CopIA Score Predictor — Data Downloader
Baixa dados históricos reais de partidas internacionais de futebol.

Fontes:
- martj42/international_results: ~47k jogos desde 1872 (GitHub)
- Ranking FIFA histórico (Kaggle / arquivo local)
- Copa 2026 schedule (arquivo local curado)
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests
from loguru import logger

# ── Caminhos ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

# ── URLs ──────────────────────────────────────────────────────────────────────
RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results"
    "/master/results.csv"
)
SHOOTOUTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results"
    "/master/shootouts.csv"
)
GOALSCORERS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results"
    "/master/goalscorers.csv"
)


def _download_csv(url: str, dest: Path, timeout: int = 30) -> pd.DataFrame:
    """Baixa CSV de uma URL com retry simples."""
    if dest.exists():
        logger.info(f"Cache encontrado: {dest.name} — pulando download")
        return pd.read_csv(dest)

    logger.info(f"Baixando {dest.name} de {url} ...")
    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(resp.content)
            df = pd.read_csv(dest)
            logger.success(f"✓ {dest.name} — {len(df):,} linhas")
            return df
        except Exception as exc:
            logger.warning(f"Tentativa {attempt+1}/3 falhou: {exc}")
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Não foi possível baixar {url}")


def download_results() -> pd.DataFrame:
    """Baixa todos os resultados internacionais (1872–hoje)."""
    return _download_csv(RESULTS_URL, RAW_DIR / "results.csv")


def download_shootouts() -> pd.DataFrame:
    """Baixa resultados de pênaltis."""
    return _download_csv(SHOOTOUTS_URL, RAW_DIR / "shootouts.csv")


def download_all() -> dict[str, pd.DataFrame]:
    """Baixa todas as fontes e retorna dicionário de DataFrames."""
    logger.info("=== Iniciando download de dados históricos ===")
    data = {
        "results": download_results(),
        "shootouts": download_shootouts(),
    }
    logger.success(f"Download concluído: {sum(len(v) for v in data.values()):,} registros totais")
    return data
