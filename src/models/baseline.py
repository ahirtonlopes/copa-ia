"""
CopIA — Modelos Baseline
Modelos de referência para comparação com abordagens mais sofisticadas.

Inclui:
1. NaiveRankingModel — sempre prediz vitória do melhor rankeado
2. PoissonModel — distribuição de Poisson bivariada (Dixon-Coles)
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import poisson


@dataclass
class MatchOutcome:
    """Probabilidades de resultado de uma partida."""
    p_win: float
    p_draw: float
    p_loss: float
    expected_goals_for: float = 1.3
    expected_goals_against: float = 1.3
    score_matrix: np.ndarray | None = None

    def most_likely_score(self) -> tuple[int, int]:
        """Retorna o placar mais provável."""
        if self.score_matrix is None:
            return (1, 1)
        idx = np.unravel_index(self.score_matrix.argmax(), self.score_matrix.shape)
        return (int(idx[0]), int(idx[1]))

    def __repr__(self):
        return (
            f"MatchOutcome(W={self.p_win:.1%}, D={self.p_draw:.1%}, L={self.p_loss:.1%}, "
            f"xG={self.expected_goals_for:.2f}:{self.expected_goals_against:.2f})"
        )


class NaiveRankingModel:
    """
    Modelo baseline: sempre prediz vitória do time melhor rankeado.
    Probabilidades derivadas apenas da diferença de ranking Elo.

    Accuracy esperada: ~55% em jogos internacionais.
    Serve como piso mínimo de performance.
    """

    def predict(self, elo_a: float, elo_b: float) -> MatchOutcome:
        """
        Prediz probabilidades usando apenas o rating Elo.
        Fórmula de Elo com ajuste para empate.
        """
        # Probabilidade base de vitória de A pelo modelo Elo
        p_win_elo = 1 / (1 + 10 ** ((elo_b - elo_a) / 400))

        # Ajuste para incluir empate (típico em futebol: ~25% dos jogos)
        # Distribuição proporcional: empate "rouba" de win e loss
        base_draw = 0.25
        p_win = p_win_elo * (1 - base_draw)
        p_loss = (1 - p_win_elo) * (1 - base_draw)
        p_draw = base_draw

        # xG aproximado pela diferença de Elo
        elo_diff = elo_a - elo_b
        lambda_a = 1.3 + elo_diff / 1000  # cada 100 Elo = +0.1 gol esperado
        lambda_b = 1.3 - elo_diff / 1000
        lambda_a = max(0.3, min(lambda_a, 4.0))
        lambda_b = max(0.3, min(lambda_b, 4.0))

        return MatchOutcome(
            p_win=p_win,
            p_draw=p_draw,
            p_loss=p_loss,
            expected_goals_for=lambda_a,
            expected_goals_against=lambda_b,
        )


class PoissonModel:
    """
    Modelo de Poisson bivariado com correção de Dixon-Coles.

    Cada time tem um parâmetro de ataque e defesa aprendido dos dados históricos.
    A taxa esperada de gols λ é calculada como:
        λ_home = ataque_home × defesa_away × fator_campo
        λ_away = ataque_away × defesa_home

    A correção de Dixon-Coles ajusta a probabilidade de placar 0-0 e 1-0/0-1,
    que são frequentemente subrepresentados pela Poisson simples.
    """

    # Parâmetros de força ofensiva/defensiva por time
    # (calibrados com dados históricos — valores estimados para 2026)
    TEAM_STRENGTH = {
        "Argentina":        {"attack": 1.82, "defense": 0.68},
        "França":           {"attack": 1.76, "defense": 0.70},
        "Brasil":           {"attack": 1.68, "defense": 0.72},
        "Espanha":          {"attack": 1.64, "defense": 0.74},
        "Inglaterra":       {"attack": 1.60, "defense": 0.76},
        "Portugal":         {"attack": 1.62, "defense": 0.78},
        "Alemanha":         {"attack": 1.58, "defense": 0.78},
        "Países Baixos":    {"attack": 1.55, "defense": 0.80},
        "Croácia":          {"attack": 1.40, "defense": 0.84},
        "Bélgica":          {"attack": 1.50, "defense": 0.82},
        "Colômbia":         {"attack": 1.48, "defense": 0.86},
        "Marrocos":         {"attack": 1.30, "defense": 0.80},
        "Uruguai":          {"attack": 1.38, "defense": 0.84},
        "Japão":            {"attack": 1.35, "defense": 0.86},
        "Senegal":          {"attack": 1.28, "defense": 0.88},
        "Dinamarca":        {"attack": 1.40, "defense": 0.84},
        "Sérvia":           {"attack": 1.35, "defense": 0.88},
        "Turquia":          {"attack": 1.30, "defense": 0.90},
        "Estados Unidos":   {"attack": 1.32, "defense": 0.90},
        "México":           {"attack": 1.28, "defense": 0.92},
        "Equador":          {"attack": 1.22, "defense": 0.94},
        "Coreia do Sul":    {"attack": 1.30, "defense": 0.90},
    }
    DEFAULT_STRENGTH = {"attack": 1.10, "defense": 1.00}

    HOME_ADVANTAGE = 1.15   # Fator de vantagem em casa (~15%)
    MAX_GOALS = 8           # Máximo de gols simulados na matriz

    def __init__(self, rho: float = -0.13):
        """
        Args:
            rho: Parâmetro de correlação Dixon-Coles (tipicamente entre -0.1 e -0.2).
                 Valores negativos corrigem a subestimação de placares baixos.
        """
        self.rho = rho

    def predict(
        self,
        team_a: str,
        team_b: str,
        neutral: bool = True,
    ) -> MatchOutcome:
        """
        Prediz probabilidades de resultado e placar mais provável.

        Args:
            team_a: Time A (home se não for campo neutro)
            team_b: Time B (away)
            neutral: Se é campo neutro (sem vantagem de casa)
        """
        str_a = self.TEAM_STRENGTH.get(team_a, self.DEFAULT_STRENGTH)
        str_b = self.TEAM_STRENGTH.get(team_b, self.DEFAULT_STRENGTH)

        home_factor = 1.0 if neutral else self.HOME_ADVANTAGE

        lambda_a = str_a["attack"] * str_b["defense"] * home_factor
        lambda_b = str_b["attack"] * str_a["defense"]

        score_matrix = self._build_score_matrix(lambda_a, lambda_b)

        p_win = float(np.tril(score_matrix, -1).sum())   # goals_a > goals_b
        p_draw = float(np.diag(score_matrix).sum())
        p_loss = float(np.triu(score_matrix, 1).sum())

        return MatchOutcome(
            p_win=p_win,
            p_draw=p_draw,
            p_loss=p_loss,
            expected_goals_for=lambda_a,
            expected_goals_against=lambda_b,
            score_matrix=score_matrix,
        )

    def predict_knockout(
        self,
        team_a: str,
        team_b: str,
    ) -> MatchOutcome:
        """
        Prediz resultado em mata-mata (prorrogação + pênaltis se empate).
        Redistribui a probabilidade de empate entre os times.
        """
        outcome = self.predict(team_a, team_b, neutral=True)

        # Em mata-mata, empate após 90min → 50/50 nos pênaltis
        # (Simplificação: poderíamos usar histórico de pênaltis por time)
        p_penalties_a = outcome.p_draw * 0.50
        p_penalties_b = outcome.p_draw * 0.50

        return MatchOutcome(
            p_win=outcome.p_win + p_penalties_a,
            p_draw=0.0,
            p_loss=outcome.p_loss + p_penalties_b,
            expected_goals_for=outcome.expected_goals_for,
            expected_goals_against=outcome.expected_goals_against,
            score_matrix=outcome.score_matrix,
        )

    def simulate_match(self, team_a: str, team_b: str, neutral: bool = True) -> tuple[int, int]:
        """
        Simula um único jogo e retorna o placar (gols_a, gols_b).
        Usado no Monte Carlo.
        """
        outcome = self.predict(team_a, team_b, neutral)
        # Amostra da distribuição de placares
        probs = outcome.score_matrix.flatten()
        idx = np.random.choice(len(probs), p=probs / probs.sum())
        g_a, g_b = divmod(idx, self.MAX_GOALS + 1)
        return int(g_a), int(g_b)

    def _build_score_matrix(self, lambda_a: float, lambda_b: float) -> np.ndarray:
        """
        Constrói a matriz de probabilidades de placar com correção Dixon-Coles.
        score_matrix[i, j] = P(goals_a=i, goals_b=j)
        """
        max_g = self.MAX_GOALS
        matrix = np.zeros((max_g + 1, max_g + 1))

        for i in range(max_g + 1):
            for j in range(max_g + 1):
                p = poisson.pmf(i, lambda_a) * poisson.pmf(j, lambda_b)
                p *= self._dixon_coles_correction(i, j, lambda_a, lambda_b)
                matrix[i, j] = p

        # Normaliza para garantir soma = 1
        matrix /= matrix.sum()
        return matrix

    def _dixon_coles_correction(
        self,
        goals_a: int,
        goals_b: int,
        lambda_a: float,
        lambda_b: float,
    ) -> float:
        """
        Fator de correção Dixon-Coles para placares baixos (0-0, 1-0, 0-1, 1-1).
        Para outros placares, o fator é 1.0.
        """
        rho = self.rho
        if goals_a == 0 and goals_b == 0:
            return 1 - lambda_a * lambda_b * rho
        elif goals_a == 1 and goals_b == 0:
            return 1 + lambda_b * rho
        elif goals_a == 0 and goals_b == 1:
            return 1 + lambda_a * rho
        elif goals_a == 1 and goals_b == 1:
            return 1 - rho
        return 1.0
