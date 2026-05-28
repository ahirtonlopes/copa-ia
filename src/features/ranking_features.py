"""
CopIA — Features de Ranking
Calcula features derivadas do ranking FIFA e rating Elo para cada seleção.
"""

import numpy as np
import pandas as pd
from loguru import logger


# Ranking FIFA aproximado para Copa 2026 (posição e pontos estimados)
# Fonte: FIFA Rankings + projeção para junho/2026
FIFA_RANKINGS_2026 = {
    "Argentina":            {"position": 1,  "points": 1872.2},
    "França":               {"position": 2,  "points": 1840.8},
    "Espanha":              {"position": 3,  "points": 1828.3},
    "Inglaterra":           {"position": 4,  "points": 1812.6},
    "Brasil":               {"position": 5,  "points": 1782.1},
    "Portugal":             {"position": 6,  "points": 1764.3},
    "Países Baixos":        {"position": 7,  "points": 1748.9},
    "Bélgica":              {"position": 8,  "points": 1736.4},
    "Alemanha":             {"position": 9,  "points": 1724.5},
    "Croácia":              {"position": 10, "points": 1712.0},
    "Itália":               {"position": 11, "points": 1698.1},
    "Colômbia":             {"position": 12, "points": 1688.2},
    "Marrocos":             {"position": 13, "points": 1678.4},
    "Uruguai":              {"position": 14, "points": 1664.5},
    "Senegal":              {"position": 15, "points": 1650.0},
    "Estados Unidos":       {"position": 16, "points": 1636.2},
    "México":               {"position": 17, "points": 1618.3},
    "Japão":                {"position": 18, "points": 1606.7},
    "Coreia do Sul":        {"position": 20, "points": 1584.5},
    "Equador":              {"position": 24, "points": 1540.1},
    "Turquia":              {"position": 26, "points": 1524.3},
    "Dinamarca":            {"position": 21, "points": 1568.0},
    "Sérvia":               {"position": 23, "points": 1548.4},
    "Canadá":               {"position": 43, "points": 1410.2},
    "Venezuela":            {"position": 50, "points": 1370.0},
    "Chile":                {"position": 30, "points": 1480.5},
    "Peru":                 {"position": 38, "points": 1430.2},
    "Nigéria":              {"position": 32, "points": 1456.7},
    "Costa do Marfim":      {"position": 36, "points": 1442.1},
    "Gana":                 {"position": 48, "points": 1388.5},
    "Camarões":             {"position": 44, "points": 1402.3},
    "África do Sul":        {"position": 52, "points": 1350.6},
    "Argélia":              {"position": 35, "points": 1448.9},
    "Arábia Saudita":       {"position": 58, "points": 1320.4},
    "Iraque":               {"position": 62, "points": 1300.1},
    "Costa Rica":           {"position": 55, "points": 1336.8},
    "Panamá":               {"position": 74, "points": 1240.5},
    "Hungria":              {"position": 45, "points": 1396.3},
    "Romênia":              {"position": 46, "points": 1392.1},
    "Eslováquia":           {"position": 40, "points": 1418.7},
    "Eslovênia":            {"position": 41, "points": 1414.5},
    "Israel":               {"position": 70, "points": 1260.8},
    "Nova Zelândia":        {"position": 96, "points": 1120.3},
    "Bolívia":              {"position": 82, "points": 1190.2},
    "Congo":                {"position": 65, "points": 1286.4},
    "Zâmbia":               {"position": 88, "points": 1152.7},
    "República Centro-Africana": {"position": 110, "points": 1060.1},
    "Iêmen":                {"position": 140, "points": 920.5},
    "Vancouver":            {"position": 99, "points": 1100.0},  # placeholder
}


def build_ranking_features(teams: list[str]) -> pd.DataFrame:
    """
    Constrói DataFrame com features de ranking para uma lista de seleções.

    Returns:
        DataFrame com índice = team, colunas = features de ranking.
    """
    rows = []
    for team in teams:
        if team not in FIFA_RANKINGS_2026:
            logger.warning(f"Ranking não encontrado para: {team}. Usando valores médios.")
            ranking_data = {"position": 80, "points": 1200.0}
        else:
            ranking_data = FIFA_RANKINGS_2026[team]

        pos = ranking_data["position"]
        pts = ranking_data["points"]

        rows.append({
            "team": team,
            "fifa_position": pos,
            "fifa_points": pts,
            # Normalizado 0–1 (maior = melhor)
            "fifa_position_norm": 1 - (pos - 1) / 200,
            "fifa_points_norm": pts / 2000,
            # Tier: 1=Elite, 2=Strong, 3=Mid, 4=Weak
            "tier": _compute_tier(pos),
            # Elo aproximado (correlacionado com pontos FIFA)
            "elo_estimated": _points_to_elo(pts),
        })

    df = pd.DataFrame(rows).set_index("team")
    return df


def compute_ranking_matchup_features(
    team_a: str,
    team_b: str,
    ranking_df: pd.DataFrame,
) -> dict:
    """
    Features derivadas do confronto de rankings entre dois times.
    Usadas como input para o modelo de predição.
    """
    a = ranking_df.loc[team_a] if team_a in ranking_df.index else _default_ranking()
    b = ranking_df.loc[team_b] if team_b in ranking_df.index else _default_ranking()

    return {
        # Diferença direta de posição (positivo = A é melhor rankeado)
        "ranking_pos_diff": b["fifa_position"] - a["fifa_position"],
        # Diferença de pontos
        "ranking_points_diff": a["fifa_points"] - b["fifa_points"],
        # Diferença de Elo estimado
        "elo_diff": a["elo_estimated"] - b["elo_estimated"],
        # Probabilidade de vitória pelo modelo Elo puro (referência)
        "elo_win_probability": _elo_win_probability(
            a["elo_estimated"], b["elo_estimated"]
        ),
        # Diferença de tier (0=igual tier, positivo=A melhor tier)
        "tier_diff": b["tier"] - a["tier"],
        # Times são do mesmo tier?
        "same_tier": int(a["tier"] == b["tier"]),
    }


def get_all_teams_ranking() -> pd.DataFrame:
    """Retorna o ranking de todos os times da Copa 2026."""
    all_teams = list(FIFA_RANKINGS_2026.keys())
    return build_ranking_features(all_teams)


# ── Funções auxiliares ─────────────────────────────────────────────────────────

def _compute_tier(position: int) -> int:
    """
    Classifica um time em tiers por ranking FIFA.
    Tier 1: Top 10 | Tier 2: 11–25 | Tier 3: 26–50 | Tier 4: 51+
    """
    if position <= 10:
        return 1
    elif position <= 25:
        return 2
    elif position <= 50:
        return 3
    return 4


def _points_to_elo(fifa_points: float) -> float:
    """
    Converte pontos FIFA em Elo aproximado.
    Calibrado empiricamente para manter a mesma escala de referência.
    """
    return 1500 + (fifa_points - 1400) * 0.4


def _elo_win_probability(elo_a: float, elo_b: float) -> float:
    """
    Probabilidade de vitória de A sobre B pelo modelo Elo puro.
    Fórmula padrão: P(A) = 1 / (1 + 10^((Elo_B - Elo_A) / 400))
    """
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))


def _default_ranking() -> dict:
    """Valores padrão para times sem dados de ranking."""
    return {
        "fifa_position": 80,
        "fifa_points": 1200.0,
        "fifa_position_norm": 0.60,
        "fifa_points_norm": 0.60,
        "tier": 4,
        "elo_estimated": 1540.0,
    }
