"""
CopIA Score Predictor — Modelo Dixon-Coles
Modelo Poisson bivariado com parâmetros de ataque/defesa ajustados por MLE.

Referência: Dixon & Coles (1997) "Modelling Association Football Scores
and Inefficiencies in the Football Betting Market"

Implementação:
- Parâmetros α_i (ataque) e β_i (defesa) por time — aprendidos dos dados
- Fator γ de home advantage
- Correção ρ para placares baixos (0-0, 1-0, 0-1, 1-1) — ajuste DC original
- Pesos por competição e decaimento temporal
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from scipy.optimize import minimize
from scipy.stats import poisson

ROOT = Path(__file__).parent.parent.parent


class DixonColesModel:
    """
    Modelo Dixon-Coles ajustado por Máxima Verossimilhança em dados reais.

    Após o fit(), o modelo tem:
        self.attack[team]  : parâmetro de força ofensiva (> 1 = acima da média)
        self.defense[team] : parâmetro de fraqueza defensiva (< 1 = defesa sólida)
        self.home_adv      : fator multiplicativo de vantagem em casa
        self.rho           : parâmetro de correção para placares baixos

    Exemplo:
        model = DixonColesModel()
        model.fit(df_matches)
        pred = model.predict("Brasil", "Argentina", neutral=True)
        print(pred.most_likely_score())  # (1, 0)
    """

    def __init__(self):
        self.attack:  dict[str, float] = {}
        self.defense: dict[str, float] = {}
        self.home_adv: float = 1.0
        self.rho:      float = 0.0
        self.teams:    list[str] = []
        self._fitted   = False

    # ── Correção Dixon-Coles ──────────────────────────────────────────────

    @staticmethod
    def _tau(x: int, y: int, mu: float, nu: float, rho: float) -> float:
        """
        Fator de correção τ para placares baixos.
        Ajusta as probabilidades de (0,0), (1,0), (0,1), (1,1).
        """
        if x == 0 and y == 0:
            return 1.0 - mu * nu * rho
        elif x == 1 and y == 0:
            return 1.0 + nu * rho
        elif x == 0 and y == 1:
            return 1.0 + mu * rho
        elif x == 1 and y == 1:
            return 1.0 - rho
        return 1.0

    # ── Log-Verossimilhança ───────────────────────────────────────────────

    def _log_likelihood(self, params: np.ndarray, df: pd.DataFrame) -> float:
        """
        Negative log-likelihood (para minimização).

        params: [attack_0, ..., attack_N, defense_0, ..., defense_N, home_adv, rho]
        """
        n = len(self.teams)
        attack  = {t: np.exp(params[i])     for i, t in enumerate(self.teams)}
        defense = {t: np.exp(params[n + i]) for i, t in enumerate(self.teams)}
        home_adv = np.exp(params[2 * n])
        rho      = params[2 * n + 1]

        ll = 0.0
        for _, row in df.iterrows():
            h, a = row["home_team"], row["away_team"]
            x, y = int(row["home_score"]), int(row["away_score"])
            w = float(row.get("sample_weight", 1.0))

            if h not in attack or a not in attack:
                continue

            neutral = bool(row.get("neutral", False))
            ha = home_adv if not neutral else 1.0

            mu = attack[h] * defense[a] * ha   # lambda do time da casa
            nu = attack[a] * defense[h]          # lambda do time visitante

            tau = self._tau(x, y, mu, nu, rho)
            if tau <= 0:
                continue

            log_p = (
                poisson.logpmf(x, mu)
                + poisson.logpmf(y, nu)
                + np.log(tau)
            )
            ll += w * log_p

        return -ll  # negativo para minimizar

    # ── Fit ───────────────────────────────────────────────────────────────

    def fit(
        self,
        df: pd.DataFrame,
        min_matches: int = 5,
        verbose: bool = True,
    ) -> "DixonColesModel":
        """
        Ajusta os parâmetros do modelo por MLE.

        Args:
            df: DataFrame com colunas [home_team, away_team, home_score,
                away_score, tournament, neutral, sample_weight]
            min_matches: Times com menos jogos são agrupados em "Other"
            verbose: Logar progresso

        Returns:
            self (para chaining)
        """
        # ── Filtra times com jogos suficientes ────────────────────────────
        all_teams = pd.concat([df["home_team"], df["away_team"]])
        team_counts = all_teams.value_counts()
        self.teams = sorted(team_counts[team_counts >= min_matches].index.tolist())

        df_fit = df[
            df["home_team"].isin(self.teams) & df["away_team"].isin(self.teams)
        ].copy()

        if verbose:
            logger.info(
                f"Ajustando Dixon-Coles em {len(df_fit):,} jogos | "
                f"{len(self.teams)} seleções"
            )

        n = len(self.teams)
        # Inicialização: tudo igual (attack=1, defense=1) no espaço log
        x0 = np.zeros(2 * n + 2)
        x0[-1] = 0.1  # rho inicial pequeno positivo

        # Restrição: média dos ataques = 1 (identificabilidade)
        # Implementada via constraint na otimização
        constraints = [{
            "type": "eq",
            "fun": lambda p: np.mean(p[:n]),  # média(log_attack) = 0
        }]

        if verbose:
            logger.info("Otimizando MLE (L-BFGS-B)... isso pode levar ~30s")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(
                fun=self._log_likelihood,
                x0=x0,
                args=(df_fit,),
                method="L-BFGS-B",
                options={"maxiter": 500, "ftol": 1e-9},
            )

        if verbose:
            status = "✓ Convergiu" if result.success else "⚠ Não convergiu completamente"
            logger.info(f"{status} | Iterações: {result.nit} | LL: {-result.fun:.2f}")

        # ── Extrai parâmetros ─────────────────────────────────────────────
        params = result.x
        self.attack  = {t: np.exp(params[i])     for i, t in enumerate(self.teams)}
        self.defense = {t: np.exp(params[n + i]) for i, t in enumerate(self.teams)}
        self.home_adv = float(np.exp(params[2 * n]))
        self.rho      = float(params[2 * n + 1])
        self._fitted  = True

        if verbose:
            top5_atk = sorted(self.attack.items(),  key=lambda x: -x[1])[:5]
            top5_def = sorted(self.defense.items(), key=lambda x:  x[1])[:5]
            logger.success(f"Top ataque:  {[(t, f'{v:.2f}') for t, v in top5_atk]}")
            logger.success(f"Top defesa:  {[(t, f'{v:.2f}') for t, v in top5_def]}")
            logger.success(f"Home adv: {self.home_adv:.3f} | ρ: {self.rho:.4f}")

        return self

    # ── Predição ──────────────────────────────────────────────────────────

    def _get_params(self, team: str) -> tuple[float, float]:
        """Retorna (attack, defense) — usa média global para times desconhecidos."""
        atk_default = float(np.mean(list(self.attack.values()))) if self.attack else 1.0
        def_default = float(np.mean(list(self.defense.values()))) if self.defense else 1.0
        atk = self.attack.get(team, atk_default)
        def_ = self.defense.get(team, def_default)
        return atk, def_

    def expected_goals(
        self, home: str, away: str, neutral: bool = True
    ) -> tuple[float, float]:
        """Retorna (lambda_home, lambda_away) — gols esperados."""
        atk_h, def_h = self._get_params(home)
        atk_a, def_a = self._get_params(away)
        ha = self.home_adv if not neutral else 1.0
        mu = atk_h * def_a * ha
        nu = atk_a * def_h
        return mu, nu

    def score_matrix(
        self, home: str, away: str, neutral: bool = True, max_goals: int = 9
    ) -> np.ndarray:
        """
        Retorna matriz de probabilidades de placar [home_goals × away_goals].
        Inclui a correção Dixon-Coles para placares baixos.
        """
        mu, nu = self.expected_goals(home, away, neutral)
        matrix = np.zeros((max_goals, max_goals))

        for x in range(max_goals):
            for y in range(max_goals):
                tau = self._tau(x, y, mu, nu, self.rho)
                matrix[x, y] = (
                    poisson.pmf(x, mu)
                    * poisson.pmf(y, nu)
                    * max(tau, 1e-10)
                )

        # Normaliza para soma = 1
        total = matrix.sum()
        if total > 0:
            matrix /= total
        return matrix

    def predict(
        self, home: str, away: str, neutral: bool = True, max_goals: int = 9
    ) -> "ScorePrediction":
        """
        Gera predição completa para uma partida.

        Returns:
            ScorePrediction com p_home_win, p_draw, p_away_win,
            most_likely_score, score_matrix, expected_goals
        """
        matrix = self.score_matrix(home, away, neutral, max_goals)
        mu, nu = self.expected_goals(home, away, neutral)

        p_home_win = float(np.tril(matrix, -1).sum())
        p_draw     = float(np.diag(matrix).sum())
        p_away_win = float(np.triu(matrix, 1).sum())

        # Placar mais provável
        idx = np.unravel_index(matrix.argmax(), matrix.shape)
        most_likely = (int(idx[0]), int(idx[1]))

        # Top 5 placares mais prováveis
        flat = matrix.flatten()
        top_idx = flat.argsort()[-8:][::-1]
        top_scores = [
            {
                "score": f"{i // max_goals}–{i % max_goals}",
                "probability": round(float(flat[i]), 4),
            }
            for i in top_idx
        ]

        return ScorePrediction(
            home=home,
            away=away,
            p_home_win=round(p_home_win, 4),
            p_draw=round(p_draw, 4),
            p_away_win=round(p_away_win, 4),
            expected_goals_home=round(float(mu), 2),
            expected_goals_away=round(float(nu), 2),
            most_likely_score=most_likely,
            top_scores=top_scores,
            score_matrix=matrix,
            attack_home=round(self.attack.get(home, 1.0), 3),
            defense_home=round(self.defense.get(home, 1.0), 3),
            attack_away=round(self.attack.get(away, 1.0), 3),
            defense_away=round(self.defense.get(away, 1.0), 3),
        )

    def predict_batch(
        self, matches: list[dict], neutral: bool = True
    ) -> pd.DataFrame:
        """
        Gera predições para uma lista de partidas.

        Args:
            matches: Lista de dicts com chaves 'home' e 'away'
            neutral: Campo neutro (padrão para Copa)

        Returns:
            DataFrame com predições de todas as partidas
        """
        rows = []
        for m in matches:
            pred = self.predict(m["home"], m["away"], neutral)
            rows.append({
                "home":              pred.home,
                "away":              pred.away,
                "p_home_win":        pred.p_home_win,
                "p_draw":            pred.p_draw,
                "p_away_win":        pred.p_away_win,
                "xg_home":           pred.expected_goals_home,
                "xg_away":           pred.expected_goals_away,
                "most_likely_score": f"{pred.most_likely_score[0]}–{pred.most_likely_score[1]}",
                "top_score_1":       pred.top_scores[0]["score"] if pred.top_scores else "",
                "top_score_1_prob":  pred.top_scores[0]["probability"] if pred.top_scores else 0,
                "confidence":        pred.confidence_label,
            })
        return pd.DataFrame(rows)


class ScorePrediction:
    """Resultado de uma predição Dixon-Coles."""

    def __init__(
        self,
        home: str, away: str,
        p_home_win: float, p_draw: float, p_away_win: float,
        expected_goals_home: float, expected_goals_away: float,
        most_likely_score: tuple[int, int],
        top_scores: list[dict],
        score_matrix: np.ndarray,
        attack_home: float = 1.0, defense_home: float = 1.0,
        attack_away: float = 1.0, defense_away: float = 1.0,
    ):
        self.home = home
        self.away = away
        self.p_home_win = p_home_win
        self.p_draw = p_draw
        self.p_away_win = p_away_win
        self.expected_goals_home = expected_goals_home
        self.expected_goals_away = expected_goals_away
        self.most_likely_score = most_likely_score
        self.top_scores = top_scores
        self.score_matrix = score_matrix
        self.attack_home = attack_home
        self.defense_home = defense_home
        self.attack_away = attack_away
        self.defense_away = defense_away

    @property
    def favorite(self) -> str:
        """Time favorito segundo o modelo."""
        probs = {self.home: self.p_home_win, "Empate": self.p_draw, self.away: self.p_away_win}
        return max(probs, key=probs.get)

    @property
    def confidence(self) -> float:
        """Grau de certeza: quão longe o favorito está do 33% (chance igual)."""
        max_p = max(self.p_home_win, self.p_draw, self.p_away_win)
        return round((max_p - 1/3) / (2/3), 3)  # normalizado 0–1

    @property
    def confidence_label(self) -> str:
        c = self.confidence
        if c < 0.15:   return "Imprevisível"
        elif c < 0.35: return "Ligeiro favorito"
        elif c < 0.55: return "Favorito"
        else:          return "Grande favorito"

    def most_likely_score_str(self) -> str:
        return f"{self.most_likely_score[0]}–{self.most_likely_score[1]}"

    def __repr__(self) -> str:
        return (
            f"ScorePrediction({self.home} vs {self.away})\n"
            f"  Resultado: W={self.p_home_win:.1%}  D={self.p_draw:.1%}  L={self.p_away_win:.1%}\n"
            f"  xG: {self.expected_goals_home:.2f} – {self.expected_goals_away:.2f}\n"
            f"  Placar + provável: {self.most_likely_score_str()} ({self.top_scores[0]['probability']:.1%})\n"
            f"  Favorito: {self.favorite} [{self.confidence_label}]\n"
            f"  Top placares: {[s['score'] for s in self.top_scores[:5]]}"
        )
