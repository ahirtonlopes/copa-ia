"""Testes do score-predictor."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.elo import EloRatingSystem, _draw_probability
from src.models.dixon_coles import DixonColesModel
from src.models.evaluator import ranked_probability_score, brier_score_multiclass, ScoreEvaluator


# ── Helpers ───────────────────────────────────────────────────────────────

def _make_matches(n: int = 200) -> pd.DataFrame:
    """Gera dataset sintético simples para testes."""
    rng = np.random.default_rng(42)
    rows = []
    teams = ["Brasil", "Argentina", "França", "Alemanha", "Espanha",
             "Inglaterra", "Portugal", "Itália", "Holanda", "Bélgica"]
    for i in range(n):
        a, b = rng.choice(teams, 2, replace=False)
        rows.append({
            "date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=int(i * 3)),
            "home_team": a, "away_team": b,
            "home_score": int(rng.poisson(1.3)),
            "away_score": int(rng.poisson(1.1)),
            "tournament": "Friendly",
            "neutral": True,
            "competition_weight": 0.3,
            "time_weight": 1.0,
            "sample_weight": 0.3,
        })
    return pd.DataFrame(rows)


# ── Testes Elo ────────────────────────────────────────────────────────────

class TestEloRatingSystem:

    def test_initial_rating(self):
        elo = EloRatingSystem()
        assert elo._get_rating("TimeSemHistórico") == 1500

    def test_winner_gains_elo(self):
        elo = EloRatingSystem()
        elo.update("Brasil", "Argentina", 2, 0, "Friendly", neutral=True)
        assert elo._get_rating("Brasil") > 1500
        assert elo._get_rating("Argentina") < 1500

    def test_elo_conservation(self):
        """Soma dos Elos deve ser conservada após qualquer jogo."""
        elo = EloRatingSystem()
        before = elo._get_rating("Brasil") + elo._get_rating("França")
        elo.update("Brasil", "França", 1, 1, "Friendly", neutral=True)
        after = elo._get_rating("Brasil") + elo._get_rating("França")
        assert abs(after - before) < 1e-6

    def test_fit_on_dataframe(self):
        df = _make_matches(100)
        elo = EloRatingSystem()
        elo.fit(df, verbose=False)
        assert len(elo.ratings) == 10  # 10 times no dataset

    def test_predict_returns_valid_probs(self):
        elo = EloRatingSystem()
        elo.fit(_make_matches(100), verbose=False)
        pred = elo.predict("Brasil", "Argentina")
        total = pred["p_home_win"] + pred["p_draw"] + pred["p_away_win"]
        assert abs(total - 1.0) < 1e-4
        for k in ("p_home_win", "p_draw", "p_away_win"):
            assert 0 <= pred[k] <= 1

    def test_home_advantage_increases_win_prob(self):
        elo = EloRatingSystem()
        pred_neutral = elo.predict("Brasil", "Argentina", neutral=True)
        pred_home    = elo.predict("Brasil", "Argentina", neutral=False)
        assert pred_home["p_home_win"] > pred_neutral["p_home_win"]

    def test_draw_probability_symmetric(self):
        assert abs(_draw_probability(0) - _draw_probability(0)) < 1e-10
        assert _draw_probability(0) > _draw_probability(200)
        assert _draw_probability(0) > _draw_probability(-200)

    def test_world_cup_k_higher_than_friendly(self):
        from src.models.elo import K_BASE
        assert K_BASE["FIFA World Cup"] > K_BASE["Friendly"]


# ── Testes Dixon-Coles ────────────────────────────────────────────────────

class TestDixonColesModel:

    def test_tau_correction_values(self):
        dc = DixonColesModel()
        # τ(0,0) deve ser < 1 quando rho > 0
        tau_00 = dc._tau(0, 0, 1.2, 1.0, 0.1)
        assert tau_00 < 1.0

        # τ para placares altos deve ser 1
        assert dc._tau(3, 2, 1.2, 1.0, 0.1) == 1.0
        assert dc._tau(2, 0, 1.2, 1.0, 0.1) == 1.0

    def test_fit_learns_team_params(self):
        df = _make_matches(200)
        dc = DixonColesModel()
        dc.fit(df, verbose=False)
        assert len(dc.attack) == len(dc.defense)
        assert len(dc.attack) > 0
        assert dc._fitted

    def test_score_matrix_sums_to_one(self):
        df = _make_matches(200)
        dc = DixonColesModel()
        dc.fit(df, verbose=False)
        matrix = dc.score_matrix("Brasil", "Argentina", neutral=True)
        assert abs(matrix.sum() - 1.0) < 1e-5

    def test_predict_probs_sum_to_one(self):
        df = _make_matches(200)
        dc = DixonColesModel()
        dc.fit(df, verbose=False)
        pred = dc.predict("Brasil", "Argentina")
        total = pred.p_home_win + pred.p_draw + pred.p_away_win
        assert abs(total - 1.0) < 1e-4

    def test_predict_unknown_team_uses_default(self):
        df = _make_matches(100)
        dc = DixonColesModel()
        dc.fit(df, verbose=False)
        # Time desconhecido deve usar média — não deve lançar erro
        pred = dc.predict("TimeMisterioso", "Brasil")
        assert pred.p_home_win + pred.p_draw + pred.p_away_win == pytest.approx(1.0, abs=1e-3)

    def test_confidence_labels(self):
        df = _make_matches(200)
        dc = DixonColesModel()
        dc.fit(df, verbose=False)
        pred = dc.predict("Brasil", "Argentina")
        assert pred.confidence_label in [
            "Imprevisível", "Ligeiro favorito", "Favorito", "Grande favorito"
        ]

    def test_predict_batch(self):
        df = _make_matches(200)
        dc = DixonColesModel()
        dc.fit(df, verbose=False)
        matches = [
            {"home": "Brasil",    "away": "Argentina"},
            {"home": "França",    "away": "Alemanha"},
            {"home": "Espanha",   "away": "Inglaterra"},
        ]
        result = dc.predict_batch(matches)
        assert len(result) == 3
        for _, row in result.iterrows():
            total = row["p_home_win"] + row["p_draw"] + row["p_away_win"]
            assert abs(total - 1.0) < 1e-3


# ── Testes Evaluator ──────────────────────────────────────────────────────

class TestEvaluator:

    def test_rps_perfect_prediction(self):
        # Prob 100% no resultado correto → RPS = 0
        rps = ranked_probability_score([1.0, 0.0, 0.0], outcome_idx=0)
        assert rps == pytest.approx(0.0, abs=1e-6)

    def test_rps_worst_prediction(self):
        # Prob 100% no resultado errado mais distante → RPS alto
        rps_wrong = ranked_probability_score([0.0, 0.0, 1.0], outcome_idx=0)
        rps_right = ranked_probability_score([1.0, 0.0, 0.0], outcome_idx=0)
        assert rps_wrong > rps_right

    def test_rps_uniform_prediction(self):
        # Predição uniforme deve ter RPS moderado
        rps = ranked_probability_score([1/3, 1/3, 1/3], outcome_idx=0)
        assert 0 < rps < 0.5

    def test_brier_score_perfect(self):
        bs = brier_score_multiclass([1.0, 0.0, 0.0], outcome_idx=0)
        assert bs == pytest.approx(0.0, abs=1e-6)

    def test_evaluator_full_pipeline(self):
        ev = ScoreEvaluator()
        ev.add(
            model_name="TestModel",
            home="Brasil", away="Argentina",
            p_home_win=0.5, p_draw=0.25, p_away_win=0.25,
            pred_score=(1, 0),
            true_home_goals=2, true_away_goals=1,
        )
        ev.add(
            model_name="TestModel",
            home="França", away="Alemanha",
            p_home_win=0.45, p_draw=0.28, p_away_win=0.27,
            pred_score=(1, 1),
            true_home_goals=1, true_away_goals=1,
        )
        result = ev.evaluate("TestModel")
        assert result.n_matches == 2
        assert 0 <= result.rps <= 1
        assert 0 <= result.brier_score <= 1
        assert 0 <= result.outcome_accuracy <= 1
