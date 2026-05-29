"""
CopIA Score Predictor — Preprocessor
Limpa e enriquece os dados históricos para treino dos modelos.

Transforma o dataset bruto em:
- matches_clean.parquet  : jogos limpos com colunas padronizadas
- matches_weighted.parquet: jogos com peso por competição e time decay
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

ROOT = Path(__file__).parent.parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

# ── Pesos por tipo de competição ───────────────────────────────────────────
# Quanto mais importante a competição, mais peso no aprendizado do modelo
COMPETITION_WEIGHTS: dict[str, float] = {
    "FIFA World Cup":            1.0,
    "FIFA World Cup qualification": 0.7,
    "UEFA Euro":                 0.85,
    "UEFA Euro qualification":   0.6,
    "Copa América":              0.85,
    "Copa America":              0.85,
    "Africa Cup of Nations":     0.75,
    "AFC Asian Cup":             0.75,
    "Gold Cup":                  0.65,
    "Confederations Cup":        0.8,
    "Nations League":            0.65,
    "Friendly":                  0.3,
    "Friendlies":                0.3,
}

# ── Decay temporal ─────────────────────────────────────────────────────────
# Jogos mais antigos têm menos peso; meia-vida ≈ 3 anos (ξ = 0.00063/dia)
DECAY_HALF_LIFE_DAYS = 3 * 365  # 3 anos


def _competition_weight(tournament: str) -> float:
    """Retorna peso da competição por nome (partial match)."""
    t = str(tournament).strip()
    for key, weight in COMPETITION_WEIGHTS.items():
        if key.lower() in t.lower():
            return weight
    # Default: tratar como qualifier de importância média
    return 0.5


def _time_decay_weight(date: pd.Timestamp, reference_date: pd.Timestamp) -> float:
    """Peso exponencial decrescente com o tempo (λ = ln(2)/half_life)."""
    days_ago = (reference_date - date).days
    if days_ago < 0:
        return 1.0
    lam = np.log(2) / DECAY_HALF_LIFE_DAYS
    return float(np.exp(-lam * days_ago))


def clean_results(df_raw: pd.DataFrame, reference_date: str | None = None) -> pd.DataFrame:
    """
    Limpa e enriquece o dataset de resultados.

    Args:
        df_raw: DataFrame bruto do download (results.csv)
        reference_date: Data de referência para decay. Default: hoje.

    Returns:
        DataFrame limpo com colunas adicionais.
    """
    df = df_raw.copy()

    # ── Tipos e parsing ───────────────────────────────────────────────────
    df["date"] = pd.to_datetime(df["date"])
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")

    # Remove jogos sem placar
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    # ── Resultado ─────────────────────────────────────────────────────────
    df["result"] = np.select(
        [df["home_score"] > df["away_score"],
         df["home_score"] < df["away_score"]],
        ["home_win", "away_win"],
        default="draw",
    )
    df["total_goals"] = df["home_score"] + df["away_score"]
    df["goal_diff"]   = df["home_score"] - df["away_score"]

    # ── Pesos ─────────────────────────────────────────────────────────────
    ref = pd.Timestamp(reference_date) if reference_date else pd.Timestamp.today()

    df["competition_weight"] = df["tournament"].apply(_competition_weight)
    df["time_weight"]        = df["date"].apply(lambda d: _time_decay_weight(d, ref))
    df["sample_weight"]      = df["competition_weight"] * df["time_weight"]

    # ── Filtros básicos ───────────────────────────────────────────────────
    # Remove jogos pré-1950 (dados muito ruidosos, poucos países)
    df = df[df["date"].dt.year >= 1950].copy()

    df = df.sort_values("date").reset_index(drop=True)

    logger.info(
        f"Dataset limpo: {len(df):,} jogos "
        f"({df['date'].dt.year.min()}–{df['date'].dt.year.max()})"
    )
    return df


def get_recent_form(
    df: pd.DataFrame,
    team: str,
    before_date: pd.Timestamp,
    n: int = 10,
) -> dict:
    """
    Calcula estatísticas de forma recente de um time antes de uma data.

    Returns:
        dict com wins, draws, losses, goals_scored_avg, goals_conceded_avg,
        goal_diff_avg, points_avg (3=W, 1=D, 0=L)
    """
    mask = (
        ((df["home_team"] == team) | (df["away_team"] == team))
        & (df["date"] < before_date)
    )
    recent = df[mask].tail(n)

    if recent.empty:
        return {
            "wins": 0, "draws": 0, "losses": 0,
            "goals_scored_avg": 1.2, "goals_conceded_avg": 1.2,
            "goal_diff_avg": 0.0, "points_avg": 1.0,
            "n_games": 0,
        }

    wins = draws = losses = 0
    goals_for = goals_against = 0

    for _, row in recent.iterrows():
        if row["home_team"] == team:
            gf, ga = row["home_score"], row["away_score"]
        else:
            gf, ga = row["away_score"], row["home_score"]

        goals_for += gf
        goals_against += ga

        if gf > ga:
            wins += 1
        elif gf == ga:
            draws += 1
        else:
            losses += 1

    n_games = len(recent)
    points_avg = (wins * 3 + draws) / n_games

    return {
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "goals_scored_avg": goals_for / n_games,
        "goals_conceded_avg": goals_against / n_games,
        "goal_diff_avg": (goals_for - goals_against) / n_games,
        "points_avg": points_avg,
        "n_games": n_games,
    }


def save_processed(df: pd.DataFrame, name: str) -> Path:
    """Salva DataFrame processado em parquet."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / f"{name}.parquet"
    df.to_parquet(path, index=False)
    logger.success(f"Salvo: {path} ({len(df):,} linhas)")
    return path


def load_processed(name: str) -> pd.DataFrame:
    """Carrega DataFrame processado do parquet."""
    path = PROCESSED_DIR / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {path}\n"
            "Execute primeiro: python -m src.data.downloader"
        )
    return pd.read_parquet(path)
