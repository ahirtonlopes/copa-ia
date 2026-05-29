"""
CopIA Score Predictor — Ensemble ML
XGBoost + LightGBM combinados em um stacking com calibração de probabilidades.

Features usadas:
- Diferença de Elo (preditor mais forte individualmente)
- Ataque/defesa Dixon-Coles dos dois times
- Forma recente (últimos 10 jogos): pontos, gols marcados/sofridos
- Dias de descanso
- Fase do torneio
- Confronto histórico (H2H)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    logger.warning("XGBoost não instalado — usando apenas LightGBM")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    logger.warning("LightGBM não instalado — usando apenas XGBoost")

ROOT = Path(__file__).parent.parent.parent

# ── Feature names ─────────────────────────────────────────────────────────
FEATURE_COLS = [
    # Elo
    "elo_home", "elo_away", "elo_diff",
    # Dixon-Coles
    "attack_home", "defense_home", "attack_away", "defense_away",
    "xg_home_dc", "xg_away_dc",
    # Forma recente (últimos 10 jogos)
    "points_avg_home", "goals_scored_avg_home", "goals_conceded_avg_home",
    "points_avg_away", "goals_scored_avg_away", "goals_conceded_avg_away",
    # Contexto
    "days_rest_home", "days_rest_away",
    "competition_weight",
    # H2H
    "h2h_win_rate_home",
    "h2h_goal_diff_avg",
]

LABEL_MAP = {"home_win": 0, "draw": 1, "away_win": 2}
LABEL_INV = {v: k for k, v in LABEL_MAP.items()}


def build_match_features(
    home: str, away: str,
    elo_system,
    dc_model,
    df_history: pd.DataFrame,
    match_date: pd.Timestamp | None = None,
    competition_weight: float = 1.0,
    neutral: bool = True,
) -> dict[str, float]:
    """
    Constrói o vetor de features para uma partida.

    Args:
        home, away: Nomes das seleções
        elo_system: EloRatingSystem fitado
        dc_model: DixonColesModel fitado
        df_history: DataFrame histórico completo (para forma e H2H)
        match_date: Data do jogo (para calcular forma antes da data)
        competition_weight: Peso da competição (1.0 = Copa do Mundo)
        neutral: Campo neutro

    Returns:
        dict mapeando feature_name → valor
    """
    from src.data.preprocessor import get_recent_form

    date = match_date or pd.Timestamp.today()

    # ── Elo ───────────────────────────────────────────────────────────────
    elo_h = elo_system._get_rating(home)
    elo_a = elo_system._get_rating(away)
    elo_diff = elo_h - elo_a + (elo_system.home_advantage if not neutral else 0)

    # ── Dixon-Coles ───────────────────────────────────────────────────────
    atk_h, def_h = dc_model._get_params(home)
    atk_a, def_a = dc_model._get_params(away)
    xg_h, xg_a  = dc_model.expected_goals(home, away, neutral)

    # ── Forma recente ─────────────────────────────────────────────────────
    form_h = get_recent_form(df_history, home, date, n=10)
    form_a = get_recent_form(df_history, away, date, n=10)

    # ── Dias de descanso (último jogo até match_date) ─────────────────────
    def days_since_last(team: str) -> float:
        mask = (df_history["home_team"] == team) | (df_history["away_team"] == team)
        past = df_history[mask & (df_history["date"] < date)]
        if past.empty:
            return 10.0  # default razoável
        last = past["date"].max()
        return min(float((date - last).days), 60.0)

    rest_h = days_since_last(home)
    rest_a = days_since_last(away)

    # ── H2H ───────────────────────────────────────────────────────────────
    h2h = df_history[
        ((df_history["home_team"] == home) & (df_history["away_team"] == away))
        | ((df_history["home_team"] == away) & (df_history["away_team"] == home))
    ]
    h2h_recent = h2h[h2h["date"] < date].tail(10)

    if h2h_recent.empty:
        h2h_win_rate = 0.5
        h2h_gd_avg   = 0.0
    else:
        wins = 0
        gd_total = 0
        for _, row in h2h_recent.iterrows():
            if row["home_team"] == home:
                gd = row["home_score"] - row["away_score"]
            else:
                gd = row["away_score"] - row["home_score"]
            gd_total += gd
            wins += 1 if gd > 0 else 0
        h2h_win_rate = wins / len(h2h_recent)
        h2h_gd_avg   = gd_total / len(h2h_recent)

    return {
        "elo_home":              elo_h,
        "elo_away":              elo_a,
        "elo_diff":              elo_diff,
        "attack_home":           atk_h,
        "defense_home":          def_h,
        "attack_away":           atk_a,
        "defense_away":          def_a,
        "xg_home_dc":            xg_h,
        "xg_away_dc":            xg_a,
        "points_avg_home":       form_h["points_avg"],
        "goals_scored_avg_home": form_h["goals_scored_avg"],
        "goals_conceded_avg_home": form_h["goals_conceded_avg"],
        "points_avg_away":       form_a["points_avg"],
        "goals_scored_avg_away": form_a["goals_scored_avg"],
        "goals_conceded_avg_away": form_a["goals_conceded_avg"],
        "days_rest_home":        rest_h,
        "days_rest_away":        rest_a,
        "competition_weight":    competition_weight,
        "h2h_win_rate_home":     h2h_win_rate,
        "h2h_goal_diff_avg":     h2h_gd_avg,
    }


class MatchEnsemble:
    """
    Ensemble de XGBoost + LightGBM para predição de resultado de partidas.
    Meta-learner: Regressão Logística calibrada.

    Prediz probabilidades de: vitória do time da casa / empate / vitória visitante.
    Compatível com qualquer seleção — usa features de Elo e DC como contexto.
    """

    def __init__(self):
        self.xgb_model: Any | None = None
        self.lgb_model: Any | None = None
        self.meta_model: Any | None = None
        self._is_fitted = False

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        eval_set: tuple | None = None,
    ) -> "MatchEnsemble":
        """
        Treina o ensemble com validação temporal.

        Args:
            X: Features (FEATURE_COLS)
            y: Labels (0=home_win, 1=draw, 2=away_win)
            eval_set: Tuple (X_val, y_val) para early stopping
        """
        logger.info(f"Treinando ensemble em {len(X):,} amostras...")

        base_preds = []

        # ── XGBoost ───────────────────────────────────────────────────────
        if HAS_XGB:
            self.xgb_model = CalibratedClassifierCV(
                xgb.XGBClassifier(
                    n_estimators=500,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="multi:softprob",
                    num_class=3,
                    eval_metric="mlogloss",
                    early_stopping_rounds=30 if eval_set else None,
                    verbosity=0,
                    random_state=42,
                ),
                cv="prefit" if eval_set else 5,
                method="isotonic",
            )
            if eval_set:
                self.xgb_model.estimator.fit(
                    X, y,
                    eval_set=[(eval_set[0], eval_set[1])],
                    verbose=False,
                )
                xgb_proba = self.xgb_model.estimator.predict_proba(X)
            else:
                self.xgb_model.fit(X, y)
                xgb_proba = self.xgb_model.predict_proba(X)
            base_preds.append(xgb_proba)
            logger.success("✓ XGBoost treinado")

        # ── LightGBM ──────────────────────────────────────────────────────
        if HAS_LGB:
            self.lgb_model = CalibratedClassifierCV(
                lgb.LGBMClassifier(
                    n_estimators=500,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="multiclass",
                    num_class=3,
                    verbosity=-1,
                    random_state=42,
                ),
                cv=5,
                method="isotonic",
            )
            self.lgb_model.fit(X, y)
            lgb_proba = self.lgb_model.predict_proba(X)
            base_preds.append(lgb_proba)
            logger.success("✓ LightGBM treinado")

        if not base_preds:
            raise RuntimeError("Nenhum modelo base disponível (instale xgboost ou lightgbm)")

        # ── Meta-learner ──────────────────────────────────────────────────
        meta_X = np.hstack(base_preds)
        self.meta_model = LogisticRegression(
            multi_class="multinomial", max_iter=500, C=1.0
        )
        self.meta_model.fit(meta_X, y)
        self._is_fitted = True
        logger.success("✓ Meta-learner (LogisticRegression) treinado")
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Retorna probabilidades [p_home_win, p_draw, p_away_win].
        Shape: (n_samples, 3)
        """
        base_preds = []
        if self.xgb_model:
            base_preds.append(self.xgb_model.predict_proba(X))
        if self.lgb_model:
            base_preds.append(self.lgb_model.predict_proba(X))
        meta_X = np.hstack(base_preds)
        return self.meta_model.predict_proba(meta_X)

    def predict_match(
        self,
        features: dict,
    ) -> dict[str, float]:
        """
        Gera predição para um jogo a partir de um dicionário de features.

        Returns:
            dict com p_home_win, p_draw, p_away_win
        """
        X = pd.DataFrame([features])[FEATURE_COLS]
        proba = self.predict_proba(X)[0]
        return {
            "p_home_win": round(float(proba[0]), 4),
            "p_draw":     round(float(proba[1]), 4),
            "p_away_win": round(float(proba[2]), 4),
        }

    def feature_importance(self) -> pd.DataFrame:
        """Retorna importância das features do XGBoost (se disponível)."""
        if not self.xgb_model:
            return pd.DataFrame()

        model = self.xgb_model.estimator if hasattr(self.xgb_model, "estimator") else self.xgb_model
        imp = model.feature_importances_
        return pd.DataFrame({
            "feature": FEATURE_COLS,
            "importance": imp,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
