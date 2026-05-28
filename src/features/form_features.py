"""
CopIA — Features de Forma Recente
Calcula performance recente de cada seleção usando janelas deslizantes
com decaimento exponencial para dar mais peso aos jogos mais recentes.
"""

import numpy as np
import pandas as pd
from loguru import logger


def compute_form_features(
    results: pd.DataFrame,
    team: str,
    reference_date: str | pd.Timestamp,
    n_games: int = 10,
    decay_factor: float = 0.85,
) -> dict:
    """
    Computa features de forma para um time em relação a uma data de referência.

    Args:
        results: DataFrame com resultados históricos (international_results.csv)
        team: Nome da seleção
        reference_date: Data de referência (ex: início da Copa)
        n_games: Número de jogos recentes a considerar
        decay_factor: Fator de decaimento exponencial (0.85 = peso 85% ao jogo anterior)

    Returns:
        Dict com todas as features de forma calculadas
    """
    ref_date = pd.Timestamp(reference_date)

    # Filtra jogos do time antes da data de referência
    mask = (
        ((results["home_team"] == team) | (results["away_team"] == team))
        & (results["date"] < ref_date)
    )
    team_games = results[mask].sort_values("date", ascending=False).head(n_games).copy()

    if len(team_games) == 0:
        logger.warning(f"Nenhum jogo encontrado para {team} antes de {reference_date}")
        return _default_form_features()

    # Determina resultado para o time (W/D/L) e gols em cada jogo
    outcomes, goals_for, goals_against, is_home = [], [], [], []

    for _, row in team_games.iterrows():
        home = row["home_team"] == team
        is_home.append(int(home))

        gf = row["home_score"] if home else row["away_score"]
        ga = row["away_score"] if home else row["home_score"]
        goals_for.append(gf)
        goals_against.append(ga)

        if gf > ga:
            outcomes.append(1)   # Vitória
        elif gf == ga:
            outcomes.append(0)   # Empate
        else:
            outcomes.append(-1)  # Derrota

    n = len(outcomes)

    # Pesos exponenciais: jogo mais recente tem peso maior
    weights = np.array([decay_factor ** i for i in range(n)])
    weights = weights / weights.sum()  # Normaliza para somar 1

    # Pontos ponderados (3=vitória, 1=empate, 0=derrota)
    points = np.array([3 if o == 1 else 1 if o == 0 else 0 for o in outcomes])
    weighted_points = float(np.dot(points, weights))

    # Gols ponderados
    gf_arr = np.array(goals_for, dtype=float)
    ga_arr = np.array(goals_against, dtype=float)
    weighted_gf = float(np.dot(gf_arr, weights))
    weighted_ga = float(np.dot(ga_arr, weights))

    # Features simples
    wins = outcomes.count(1)
    draws = outcomes.count(0)
    losses = outcomes.count(-1)

    return {
        # Contagens diretas (últimos N jogos)
        "form_games_played": n,
        "form_wins": wins,
        "form_draws": draws,
        "form_losses": losses,
        "form_win_rate": wins / n,
        "form_draw_rate": draws / n,
        "form_loss_rate": losses / n,

        # Gols
        "form_goals_scored": sum(goals_for),
        "form_goals_conceded": sum(goals_against),
        "form_goal_diff": sum(goals_for) - sum(goals_against),
        "form_goals_per_game": sum(goals_for) / n,
        "form_conceded_per_game": sum(goals_against) / n,

        # Versões ponderadas por recência (mais informativas)
        "form_weighted_points": weighted_points,
        "form_weighted_goals_scored": weighted_gf,
        "form_weighted_goals_conceded": weighted_ga,
        "form_weighted_goal_diff": weighted_gf - weighted_ga,

        # Clean sheets
        "form_clean_sheets": sum(1 for ga in goals_against if ga == 0),
        "form_clean_sheet_rate": sum(1 for ga in goals_against if ga == 0) / n,

        # Tendência (últimos 5 vs. 5 anteriores — precisa de n >= 10)
        "form_trend": _compute_trend(points, n),

        # Contexto
        "form_home_pct": sum(is_home) / n,
    }


def build_form_matrix(
    results: pd.DataFrame,
    teams: list[str],
    reference_date: str,
    n_games: int = 10,
) -> pd.DataFrame:
    """
    Constrói matrix de features de forma para todos os times.

    Returns:
        DataFrame com índice = team, colunas = features de forma.
    """
    rows = {}
    for team in teams:
        features = compute_form_features(results, team, reference_date, n_games)
        rows[team] = features

    return pd.DataFrame(rows).T


def compute_form_matchup_features(
    form_a: dict,
    form_b: dict,
) -> dict:
    """
    Features derivadas da comparação de forma entre dois times.
    Retorna diferenças relativas (A - B) para uso no modelo de predição.
    """
    keys_to_compare = [
        "form_weighted_points",
        "form_win_rate",
        "form_goals_per_game",
        "form_conceded_per_game",
        "form_weighted_goal_diff",
        "form_clean_sheet_rate",
        "form_trend",
    ]

    matchup = {}
    for key in keys_to_compare:
        val_a = form_a.get(key, 0)
        val_b = form_b.get(key, 0)
        # Diferença: positivo = A está em melhor forma
        matchup[f"form_diff_{key.replace('form_', '')}"] = val_a - val_b

    # Feature composta: quem está em melhor forma geral
    matchup["form_advantage"] = (
        form_a.get("form_weighted_points", 1.5)
        - form_b.get("form_weighted_points", 1.5)
    )

    return matchup


# ── Funções auxiliares ─────────────────────────────────────────────────────────

def _compute_trend(points: np.ndarray, n: int) -> float:
    """
    Compara forma da primeira metade vs. segunda metade dos últimos N jogos.
    Retorna: positivo = melhorando, negativo = piorando, 0 = estável.
    Note: array está em ordem decrescente (mais recente primeiro).
    """
    if n < 4:
        return 0.0
    half = n // 2
    recent = points[:half].mean()    # Mais recentes
    older = points[half:].mean()     # Mais antigos
    return float(recent - older)


def _default_form_features() -> dict:
    """Valores neutros para times sem histórico suficiente."""
    return {
        "form_games_played": 0,
        "form_wins": 0,
        "form_draws": 0,
        "form_losses": 0,
        "form_win_rate": 0.33,
        "form_draw_rate": 0.33,
        "form_loss_rate": 0.33,
        "form_goals_scored": 0,
        "form_goals_conceded": 0,
        "form_goal_diff": 0,
        "form_goals_per_game": 1.2,
        "form_conceded_per_game": 1.2,
        "form_weighted_points": 1.5,
        "form_weighted_goals_scored": 1.2,
        "form_weighted_goals_conceded": 1.2,
        "form_weighted_goal_diff": 0.0,
        "form_clean_sheets": 0,
        "form_clean_sheet_rate": 0.2,
        "form_trend": 0.0,
        "form_home_pct": 0.5,
    }
