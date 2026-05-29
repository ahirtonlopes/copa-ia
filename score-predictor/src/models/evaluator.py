"""
CopIA Score Predictor — Avaliação de Modelos
Métricas padrão da indústria para modelos probabilísticos de futebol.

Métricas implementadas:
- RPS  : Ranked Probability Score (padrão sports analytics)
- Brier: Brier Score por classe
- Acurácia de resultado (W/D/L)
- Acerto de placar exato
- MAE de gols (home e away separados)
- Comparação entre modelos (tabela comparativa)
- Evolução ao longo da Copa (curva de RPS por rodada)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class EvalResult:
    """Resultado de avaliação de um modelo em um conjunto de jogos."""
    model_name: str
    n_matches: int

    # Métricas principais
    rps: float = 0.0              # Ranked Probability Score (menor = melhor)
    brier_score: float = 0.0      # Brier Score médio
    outcome_accuracy: float = 0.0 # % de resultados corretos (W/D/L)
    exact_score_accuracy: float = 0.0  # % de placares exatos corretos
    mae_goals_home: float = 0.0   # MAE no número de gols do time da casa
    mae_goals_away: float = 0.0   # MAE no número de gols do visitante
    mae_total_goals: float = 0.0  # MAE no total de gols

    # Detalhe por resultado
    accuracy_by_outcome: dict = field(default_factory=dict)
    calibration_bins: list = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"\n{'='*50}\n"
            f"  {self.model_name} — {self.n_matches} jogos\n"
            f"{'='*50}\n"
            f"  RPS (↓):            {self.rps:.4f}\n"
            f"  Brier Score (↓):    {self.brier_score:.4f}\n"
            f"  Acurácia resultado: {self.outcome_accuracy:.1%}\n"
            f"  Acerto placar exato:{self.exact_score_accuracy:.1%}\n"
            f"  MAE gols (casa):    {self.mae_goals_home:.2f}\n"
            f"  MAE gols (fora):    {self.mae_goals_away:.2f}\n"
            f"  MAE total gols:     {self.mae_total_goals:.2f}\n"
        )


def ranked_probability_score(
    probs: list[float], outcome_idx: int, n_classes: int = 3
) -> float:
    """
    Ranked Probability Score para uma única predição.

    RPS é a métrica padrão para modelos probabilísticos de futebol.
    Penaliza mais predições que erram por mais categorias.

    Args:
        probs: [p_home_win, p_draw, p_away_win]
        outcome_idx: 0=home_win, 1=draw, 2=away_win
        n_classes: Número de classes (3 para futebol)

    Returns:
        RPS para esta predição (0 = perfeito, 1 = pior possível)
    """
    cum_probs = np.cumsum(probs)
    cum_true = np.zeros(n_classes)
    cum_true[outcome_idx:] = 1.0
    cum_true = np.cumsum(np.eye(n_classes)[outcome_idx])

    # Acumula as probabilidades verdadeiras
    true_cumsum = np.zeros(n_classes - 1)
    for i in range(n_classes - 1):
        true_cumsum[i] = 1.0 if outcome_idx <= i else 0.0

    pred_cumsum = np.cumsum(probs)[:-1]
    rps = np.mean((pred_cumsum - true_cumsum) ** 2)
    return float(rps)


def brier_score_multiclass(
    probs: list[float], outcome_idx: int
) -> float:
    """Brier Score multiclasse para uma predição."""
    one_hot = np.zeros(len(probs))
    one_hot[outcome_idx] = 1.0
    return float(np.mean((np.array(probs) - one_hot) ** 2))


class ScoreEvaluator:
    """
    Avalia e compara modelos de predição de placares.

    Uso:
        evaluator = ScoreEvaluator()

        # Adiciona predições e resultados reais
        evaluator.add(
            model_name="Dixon-Coles",
            home="Brasil", away="Argentina",
            p_home_win=0.45, p_draw=0.27, p_away_win=0.28,
            pred_score=(1, 0),
            true_home_goals=2, true_away_goals=1,
        )

        # Avalia
        result = evaluator.evaluate("Dixon-Coles")
        print(result)

        # Compara todos
        evaluator.compare_models()
    """

    def __init__(self):
        self._records: list[dict] = []

    def add(
        self,
        model_name: str,
        home: str,
        away: str,
        p_home_win: float,
        p_draw: float,
        p_away_win: float,
        pred_score: tuple[int, int],
        true_home_goals: int,
        true_away_goals: int,
        match_date: str | None = None,
        round_name: str | None = None,
    ) -> None:
        """
        Registra uma predição com seu resultado real.

        Args:
            model_name: Identificador do modelo
            home, away: Times
            p_home_win, p_draw, p_away_win: Probabilidades preditas
            pred_score: Placar predito (mais provável)
            true_home_goals, true_away_goals: Placar real
            match_date: Data do jogo
            round_name: Fase (Grupo A, Oitavas, etc.)
        """
        # Resultado real
        if true_home_goals > true_away_goals:
            true_outcome = "home_win"
            true_idx = 0
        elif true_home_goals == true_away_goals:
            true_outcome = "draw"
            true_idx = 1
        else:
            true_outcome = "away_win"
            true_idx = 2

        probs = [p_home_win, p_draw, p_away_win]
        pred_outcome_idx = int(np.argmax(probs))
        pred_outcomes = ["home_win", "draw", "away_win"]

        self._records.append({
            "model":            model_name,
            "home":             home,
            "away":             away,
            "p_home_win":       p_home_win,
            "p_draw":           p_draw,
            "p_away_win":       p_away_win,
            "pred_score_home":  pred_score[0],
            "pred_score_away":  pred_score[1],
            "true_home_goals":  true_home_goals,
            "true_away_goals":  true_away_goals,
            "true_outcome":     true_outcome,
            "true_outcome_idx": true_idx,
            "pred_outcome":     pred_outcomes[pred_outcome_idx],
            "outcome_correct":  pred_outcomes[pred_outcome_idx] == true_outcome,
            "score_exact":      (pred_score[0] == true_home_goals and
                                 pred_score[1] == true_away_goals),
            "rps":              ranked_probability_score(probs, true_idx),
            "brier":            brier_score_multiclass(probs, true_idx),
            "match_date":       match_date,
            "round":            round_name,
        })

    def evaluate(self, model_name: str) -> EvalResult:
        """Calcula métricas agregadas para um modelo."""
        records = [r for r in self._records if r["model"] == model_name]
        if not records:
            raise ValueError(f"Modelo '{model_name}' não encontrado nos registros")

        df = pd.DataFrame(records)
        n = len(df)

        # MAE de gols
        mae_h = float(np.mean(np.abs(df["pred_score_home"] - df["true_home_goals"])))
        mae_a = float(np.mean(np.abs(df["pred_score_away"] - df["true_away_goals"])))
        mae_t = float(np.mean(
            np.abs((df["pred_score_home"] + df["pred_score_away"])
                   - (df["true_home_goals"] + df["true_away_goals"]))
        ))

        # Acurácia por tipo de resultado
        acc_by_outcome = df.groupby("true_outcome")["outcome_correct"].mean().to_dict()

        return EvalResult(
            model_name=model_name,
            n_matches=n,
            rps=float(df["rps"].mean()),
            brier_score=float(df["brier"].mean()),
            outcome_accuracy=float(df["outcome_correct"].mean()),
            exact_score_accuracy=float(df["score_exact"].mean()),
            mae_goals_home=mae_h,
            mae_goals_away=mae_a,
            mae_total_goals=mae_t,
            accuracy_by_outcome=acc_by_outcome,
        )

    def compare_models(self) -> pd.DataFrame:
        """
        Tabela comparativa de todos os modelos registrados.
        Ordena por RPS (menor = melhor).
        """
        models = list({r["model"] for r in self._records})
        rows = []
        for m in models:
            ev = self.evaluate(m)
            rows.append({
                "Modelo":         ev.model_name,
                "Jogos":          ev.n_matches,
                "RPS ↓":          f"{ev.rps:.4f}",
                "Brier ↓":        f"{ev.brier_score:.4f}",
                "Acurácia W/D/L": f"{ev.outcome_accuracy:.1%}",
                "Placar exato":   f"{ev.exact_score_accuracy:.1%}",
                "MAE gols":       f"{ev.mae_total_goals:.2f}",
            })

        df = pd.DataFrame(rows).sort_values("RPS ↓")
        print("\n" + df.to_string(index=False))
        return df

    def rps_by_round(self, model_name: str) -> pd.DataFrame:
        """
        RPS médio por rodada — útil para visualizar evolução durante a Copa.
        """
        records = [r for r in self._records if r["model"] == model_name and r["round"]]
        if not records:
            return pd.DataFrame()
        df = pd.DataFrame(records)
        return df.groupby("round")["rps"].mean().reset_index()

    def to_dataframe(self) -> pd.DataFrame:
        """Retorna todos os registros como DataFrame."""
        return pd.DataFrame(self._records)


def evaluate_backtesting(
    predictions_df: pd.DataFrame,
    results_df: pd.DataFrame,
    model_name: str = "Model",
) -> EvalResult:
    """
    Avalia um modelo em modo backtesting, mesclando predições com resultados reais.

    Args:
        predictions_df: DataFrame com colunas [home, away, p_home_win, p_draw,
                        p_away_win, most_likely_score, date]
        results_df: DataFrame com resultados reais [home_team, away_team,
                   home_score, away_score, date]
        model_name: Nome do modelo para o relatório

    Returns:
        EvalResult com todas as métricas
    """
    evaluator = ScoreEvaluator()

    merged = predictions_df.merge(
        results_df,
        left_on=["home", "away", "date"],
        right_on=["home_team", "away_team", "date"],
        how="inner",
    )

    if merged.empty:
        logger.warning("Nenhuma predição correspondeu a resultados reais")
        return EvalResult(model_name=model_name, n_matches=0)

    for _, row in merged.iterrows():
        score_str = str(row.get("most_likely_score", "1-0"))
        parts = score_str.replace("–", "-").split("-")
        pred_h = int(parts[0]) if len(parts) > 0 else 1
        pred_a = int(parts[1]) if len(parts) > 1 else 0

        evaluator.add(
            model_name=model_name,
            home=row["home"],
            away=row["away"],
            p_home_win=float(row["p_home_win"]),
            p_draw=float(row["p_draw"]),
            p_away_win=float(row["p_away_win"]),
            pred_score=(pred_h, pred_a),
            true_home_goals=int(row["home_score"]),
            true_away_goals=int(row["away_score"]),
            match_date=str(row["date"]),
        )

    result = evaluator.evaluate(model_name)
    logger.info(result)
    return result
