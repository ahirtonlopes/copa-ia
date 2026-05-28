"""
CopIA — Features de Head-to-Head
Calcula o histórico de confrontos diretos entre dois times.
"""

import numpy as np
import pandas as pd
from loguru import logger


def compute_h2h_features(
    results: pd.DataFrame,
    team_a: str,
    team_b: str,
    reference_date: str | pd.Timestamp | None = None,
    max_games: int = 20,
) -> dict:
    """
    Calcula features do histórico de confrontos diretos entre dois times.

    Args:
        results: DataFrame de resultados históricos
        team_a: Primeira seleção
        team_b: Segunda seleção
        reference_date: Considera apenas jogos antes dessa data
        max_games: Máximo de jogos H2H a considerar

    Returns:
        Dict com features H2H do ponto de vista de team_a vs. team_b
    """
    ref_date = pd.Timestamp(reference_date) if reference_date else pd.Timestamp.now()

    # Filtra confrontos diretos entre os dois times
    mask = (
        (
            ((results["home_team"] == team_a) & (results["away_team"] == team_b))
            | ((results["home_team"] == team_b) & (results["away_team"] == team_a))
        )
        & (results["date"] < ref_date)
    )
    h2h_games = results[mask].sort_values("date", ascending=False).head(max_games).copy()

    n = len(h2h_games)

    if n == 0:
        logger.debug(f"Nenhum H2H encontrado: {team_a} vs {team_b}")
        return _default_h2h_features()

    # Calcula resultado do ponto de vista de team_a
    wins_a, draws, wins_b = 0, 0, 0
    goals_a, goals_b = [], []

    for _, row in h2h_games.iterrows():
        a_is_home = row["home_team"] == team_a
        gf = row["home_score"] if a_is_home else row["away_score"]
        ga = row["away_score"] if a_is_home else row["home_score"]

        goals_a.append(gf)
        goals_b.append(ga)

        if gf > ga:
            wins_a += 1
        elif gf == ga:
            draws += 1
        else:
            wins_b += 1

    total_goals_a = sum(goals_a)
    total_goals_b = sum(goals_b)

    # Recência: peso maior para jogos mais recentes
    weights = np.array([0.85 ** i for i in range(n)])
    weights /= weights.sum()

    outcomes = np.array([
        1 if ga > gb else 0 if ga == gb else -1
        for ga, gb in zip(goals_a, goals_b)
    ])
    weighted_form = float(np.dot(outcomes, weights))

    # Última vez que se enfrentaram
    last_game = h2h_games.iloc[0]
    last_result = (
        "win" if (
            (last_game["home_team"] == team_a and last_game["home_score"] > last_game["away_score"])
            or (last_game["away_team"] == team_a and last_game["away_score"] > last_game["home_score"])
        )
        else "draw" if last_game["home_score"] == last_game["away_score"]
        else "loss"
    )

    return {
        # Contagens H2H completo
        "h2h_total_games": n,
        "h2h_wins_a": wins_a,
        "h2h_draws": draws,
        "h2h_wins_b": wins_b,

        # Taxas
        "h2h_win_rate_a": wins_a / n,
        "h2h_draw_rate": draws / n,
        "h2h_win_rate_b": wins_b / n,

        # Gols
        "h2h_goals_a": total_goals_a,
        "h2h_goals_b": total_goals_b,
        "h2h_goal_diff_a": total_goals_a - total_goals_b,
        "h2h_goals_per_game": (total_goals_a + total_goals_b) / n,

        # Forma ponderada recente no H2H
        "h2h_weighted_form": weighted_form,

        # Informação qualitativa
        "h2h_last_result": last_result,
        "h2h_last_result_encoded": 1 if last_result == "win" else 0 if last_result == "draw" else -1,

        # Dominância histórica
        "h2h_dominance": _compute_dominance(wins_a, draws, wins_b),
    }


def build_h2h_matrix(
    results: pd.DataFrame,
    teams: list[str],
    reference_date: str,
) -> dict[tuple, dict]:
    """
    Pré-computa todos os confrontos H2H entre os times da Copa.
    Retorna dict com chave (team_a, team_b) e valor = features H2H.
    """
    h2h_cache = {}
    pairs_computed = 0

    for i, team_a in enumerate(teams):
        for team_b in teams[i + 1:]:
            features = compute_h2h_features(results, team_a, team_b, reference_date)
            h2h_cache[(team_a, team_b)] = features
            # Perspectiva inversa (simétrica com sinais invertidos)
            h2h_cache[(team_b, team_a)] = _invert_h2h(features)
            pairs_computed += 1

    logger.info(f"H2H pré-computado para {pairs_computed} pares de times")
    return h2h_cache


# ── Funções auxiliares ─────────────────────────────────────────────────────────

def _compute_dominance(wins_a: int, draws: int, wins_b: int) -> float:
    """
    Índice de dominância de A sobre B: -1 (B domina total) a +1 (A domina total).
    """
    total = wins_a + draws + wins_b
    if total == 0:
        return 0.0
    return (wins_a - wins_b) / total


def _invert_h2h(features: dict) -> dict:
    """Inverte a perspectiva de A para B em features H2H."""
    inv = features.copy()
    inv["h2h_wins_a"], inv["h2h_wins_b"] = features["h2h_wins_b"], features["h2h_wins_a"]
    inv["h2h_win_rate_a"], inv["h2h_win_rate_b"] = features["h2h_win_rate_b"], features["h2h_win_rate_a"]
    inv["h2h_goals_a"], inv["h2h_goals_b"] = features["h2h_goals_b"], features["h2h_goals_a"]
    inv["h2h_goal_diff_a"] = -features["h2h_goal_diff_a"]
    inv["h2h_weighted_form"] = -features["h2h_weighted_form"]
    inv["h2h_dominance"] = -features["h2h_dominance"]
    inv["h2h_last_result_encoded"] = -features["h2h_last_result_encoded"]
    last = features["h2h_last_result"]
    inv["h2h_last_result"] = "win" if last == "loss" else "loss" if last == "win" else "draw"
    return inv


def _default_h2h_features() -> dict:
    """Features neutras para confrontos sem histórico."""
    return {
        "h2h_total_games": 0,
        "h2h_wins_a": 0,
        "h2h_draws": 0,
        "h2h_wins_b": 0,
        "h2h_win_rate_a": 0.33,
        "h2h_draw_rate": 0.33,
        "h2h_win_rate_b": 0.33,
        "h2h_goals_a": 0,
        "h2h_goals_b": 0,
        "h2h_goal_diff_a": 0,
        "h2h_goals_per_game": 2.5,
        "h2h_weighted_form": 0.0,
        "h2h_last_result": "unknown",
        "h2h_last_result_encoded": 0,
        "h2h_dominance": 0.0,
    }
