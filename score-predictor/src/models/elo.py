"""
CopIA Score Predictor — Sistema de Rating Elo Dinâmico
Implementação do Elo estilo World Football Elo Ratings (eloratings.net)
com K-factor por competição, ajuste de placar e cálculo de probabilidades.

Referência: https://eloratings.net/about
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from loguru import logger

ROOT = Path(__file__).parent.parent.parent

# ── K-factors por competição ───────────────────────────────────────────────
# Baseado no sistema oficial da FIFA e World Football Elo
K_BASE: dict[str, int] = {
    "FIFA World Cup":                 60,
    "FIFA World Cup qualification":   40,
    "UEFA Euro":                      50,
    "UEFA Euro qualification":        35,
    "Copa América":                   50,
    "Copa America":                   50,
    "Africa Cup of Nations":          45,
    "AFC Asian Cup":                  45,
    "Gold Cup":                       35,
    "Confederations Cup":             45,
    "Nations League":                 35,
    "Friendly":                       20,
    "Friendlies":                     20,
}

# Elo inicial para times sem histórico
ELO_DEFAULT = 1500
ELO_START   = 1500  # Elo inicial padrão


class MatchResult(NamedTuple):
    """Resultado de uma partida com atualização de Elo."""
    team_a: str
    team_b: str
    elo_a_before: float
    elo_b_before: float
    elo_a_after:  float
    elo_b_after:  float
    expected_a: float   # Probabilidade esperada para A vencer
    actual_a: float     # Resultado real (1=vitória A, 0.5=empate, 0=derrota A)
    score_a: int
    score_b: int


class EloRatingSystem:
    """
    Sistema de rating Elo para seleções de futebol.

    Features:
    - K-factor dinâmico por tipo de competição
    - Ajuste de placar (goal difference multiplier)
    - Campo neutro vs. home advantage
    - Histórico completo de mudanças de rating
    - Probabilidades de vitória/empate/derrota via Elo diff

    Exemplo:
        elo = EloRatingSystem()
        elo.fit(df_matches)
        probs = elo.predict("Brasil", "Argentina")
        print(probs)  # {"p_home_win": 0.45, "p_draw": 0.28, "p_away_win": 0.27}
    """

    # Vantagem de campo em pontos Elo (equivale a ~0.04 em prob)
    HOME_ADVANTAGE: float = 100.0

    def __init__(self, home_advantage: float = HOME_ADVANTAGE):
        self.ratings: dict[str, float] = {}
        self.history: list[MatchResult] = []
        self.home_advantage = home_advantage
        self._n_matches_processed = 0

    # ── Internos ──────────────────────────────────────────────────────────

    def _get_rating(self, team: str) -> float:
        return self.ratings.get(team, ELO_DEFAULT)

    @staticmethod
    def _k_factor(tournament: str, goal_diff: int) -> float:
        """
        K-factor = base_k × multiplicador de placar.
        Placar elástico: goleadas valem mais.
        """
        base_k = K_BASE.get(tournament, 40)

        # Multiplicador de placar (World Football Elo standard)
        if goal_diff == 0:
            gd_mult = 1.0
        elif goal_diff == 1:
            gd_mult = 1.0
        elif goal_diff == 2:
            gd_mult = 1.5
        else:
            gd_mult = (11 + goal_diff) / 8.0

        return base_k * gd_mult

    @staticmethod
    def _k_factor_by_tournament(tournament: str) -> float:
        """K sem multiplicador de placar (para estimativas)."""
        t = str(tournament).lower()
        for key, k in K_BASE.items():
            if key.lower() in t:
                return float(k)
        return 40.0

    @staticmethod
    def _expected_score(elo_a: float, elo_b: float) -> float:
        """Probabilidade esperada de vitória de A pelo modelo Elo puro."""
        return 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))

    def _actual_score(self, score_a: int, score_b: int) -> float:
        """Converte placar em resultado Elo (1, 0.5, 0)."""
        if score_a > score_b:
            return 1.0
        elif score_a < score_b:
            return 0.0
        return 0.5

    # ── API pública ───────────────────────────────────────────────────────

    def update(
        self,
        team_a: str,
        team_b: str,
        score_a: int,
        score_b: int,
        tournament: str = "Friendly",
        neutral: bool = False,
    ) -> MatchResult:
        """
        Processa um jogo e atualiza os ratings.

        Args:
            team_a: Time da casa (ou time A em campo neutro)
            team_b: Time visitante
            score_a: Gols do time A
            score_b: Gols do time B
            tournament: Nome da competição
            neutral: True = campo neutro, False = time A joga em casa

        Returns:
            MatchResult com ratings antes/depois e probabilidades
        """
        elo_a = self._get_rating(team_a)
        elo_b = self._get_rating(team_b)

        # Ajuste de home advantage (apenas em jogos não neutros)
        elo_a_adj = elo_a + (self.home_advantage if not neutral else 0)

        expected_a = self._expected_score(elo_a_adj, elo_b)
        actual_a   = self._actual_score(score_a, score_b)

        goal_diff  = abs(score_a - score_b)
        k          = self._k_factor(tournament, goal_diff)

        delta = k * (actual_a - expected_a)

        new_elo_a = elo_a + delta
        new_elo_b = elo_b - delta

        self.ratings[team_a] = new_elo_a
        self.ratings[team_b] = new_elo_b

        result = MatchResult(
            team_a=team_a, team_b=team_b,
            elo_a_before=elo_a, elo_b_before=elo_b,
            elo_a_after=new_elo_a, elo_b_after=new_elo_b,
            expected_a=expected_a, actual_a=actual_a,
            score_a=score_a, score_b=score_b,
        )
        self.history.append(result)
        self._n_matches_processed += 1
        return result

    def fit(self, df: pd.DataFrame, verbose: bool = True) -> "EloRatingSystem":
        """
        Treina o sistema Elo em um histórico completo de jogos.

        Args:
            df: DataFrame com colunas [date, home_team, away_team,
                home_score, away_score, tournament, neutral]
            verbose: Logar progresso
        """
        df_sorted = df.sort_values("date")
        total = len(df_sorted)

        for i, row in df_sorted.iterrows():
            self.update(
                team_a=row["home_team"],
                team_b=row["away_team"],
                score_a=int(row["home_score"]),
                score_b=int(row["away_score"]),
                tournament=str(row.get("tournament", "Friendly")),
                neutral=bool(row.get("neutral", False)),
            )

        if verbose:
            logger.success(
                f"Elo ajustado em {total:,} jogos | "
                f"{len(self.ratings)} times | "
                f"Top Elo: {self.top_teams(3)}"
            )
        return self

    def predict(
        self,
        team_a: str,
        team_b: str,
        neutral: bool = True,
    ) -> dict[str, float]:
        """
        Prevê probabilidades para um confronto.

        Usa o modelo de conversão Elo→prob baseado em dados históricos
        da World Football Elo (inclui probabilidade de empate implícita).

        Returns:
            dict com p_home_win, p_draw, p_away_win, elo_diff,
                      elo_home, elo_away, expected_goals_home, expected_goals_away
        """
        elo_a = self._get_rating(team_a)
        elo_b = self._get_rating(team_b)

        elo_a_adj = elo_a + (self.home_advantage if not neutral else 0)
        elo_diff  = elo_a_adj - elo_b

        # Prob de vitória puro (sem empate)
        p_win_pure = self._expected_score(elo_a_adj, elo_b)

        # Distribuição win/draw/loss calibrada empiricamente
        # Baseada em conversão padrão da World Football Elo
        p_draw = _draw_probability(elo_diff)
        p_win  = p_win_pure  * (1 - p_draw)
        p_loss = (1 - p_win_pure) * (1 - p_draw)

        # xG esperado (aproximação via Elo diff — calibrada em dados históricos)
        avg_goals = 1.15  # média de gols em jogos internacionais neutros
        lambda_a = avg_goals * np.exp(elo_diff / 1200)
        lambda_b = avg_goals * np.exp(-elo_diff / 1200)

        return {
            "team_home": team_a,
            "team_away": team_b,
            "elo_home": round(elo_a, 1),
            "elo_away": round(elo_b, 1),
            "elo_diff": round(elo_diff, 1),
            "p_home_win": round(p_win,  4),
            "p_draw":     round(p_draw, 4),
            "p_away_win": round(p_loss, 4),
            "expected_goals_home": round(float(lambda_a), 2),
            "expected_goals_away": round(float(lambda_b), 2),
        }

    def top_teams(self, n: int = 20) -> list[tuple[str, float]]:
        """Retorna os N times com maior rating."""
        return sorted(self.ratings.items(), key=lambda x: -x[1])[:n]

    def rating_table(self, n: int = 32) -> pd.DataFrame:
        """Retorna DataFrame com ranking dos times."""
        teams = self.top_teams(n)
        return pd.DataFrame(teams, columns=["team", "elo"]).assign(
            rank=lambda df: range(1, len(df) + 1)
        )[["rank", "team", "elo"]]

    def save(self, path: Path | None = None) -> Path:
        """Persiste ratings em JSON."""
        path = path or ROOT / "outputs" / "models" / "elo_ratings.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(
                {"ratings": self.ratings, "n_matches": self._n_matches_processed},
                f, indent=2, ensure_ascii=False,
            )
        logger.success(f"Ratings Elo salvos em {path}")
        return path

    @classmethod
    def load(cls, path: Path) -> "EloRatingSystem":
        """Carrega ratings de JSON."""
        with open(path) as f:
            data = json.load(f)
        elo = cls()
        elo.ratings = data["ratings"]
        elo._n_matches_processed = data.get("n_matches", 0)
        logger.info(f"Elo carregado: {len(elo.ratings)} times")
        return elo


# ── Helpers ───────────────────────────────────────────────────────────────

def _draw_probability(elo_diff: float) -> float:
    """
    Probabilidade de empate como função da diferença de Elo.
    Empates são mais prováveis quando times são equilibrados.
    Calibrada empiricamente em dados de futebol internacional.
    """
    # Pico de ~28% em partidas equilibradas (elo_diff ≈ 0)
    # Decai à medida que o desequilíbrio aumenta
    base_draw = 0.28
    decay = 0.0004
    return base_draw * np.exp(-decay * elo_diff ** 2)
