"""
CopIA — Testes dos Modelos Baseline e Features
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.baseline import NaiveRankingModel, PoissonModel, MatchOutcome
from src.features.ranking_features import (
    build_ranking_features,
    compute_ranking_matchup_features,
    get_all_teams_ranking,
)
from src.features.form_features import compute_form_features
from src.features.h2h_features import compute_h2h_features


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_results():
    """Dataset mínimo de resultados para testes."""
    return pd.DataFrame({
        "date": pd.to_datetime([
            "2024-01-10", "2024-02-15", "2024-03-20",
            "2024-04-10", "2024-05-05", "2024-06-01",
            "2024-07-15", "2024-08-20", "2024-09-10",
            "2024-10-05",
        ]),
        "home_team": [
            "Brasil", "Argentina", "Brasil", "Argentina",
            "França", "Alemanha", "Brasil", "Argentina",
            "Brasil", "França",
        ],
        "away_team": [
            "Argentina", "França", "Alemanha", "Espanha",
            "Espanha", "Brasil", "França", "Alemanha",
            "Espanha", "Alemanha",
        ],
        "home_score": [2, 1, 1, 2, 0, 1, 3, 1, 2, 1],
        "away_score": [1, 0, 1, 1, 0, 2, 1, 1, 0, 1],
        "tournament": ["Friendly"] * 10,
        "neutral": [False] * 10,
    })


# ── Testes NaiveRankingModel ───────────────────────────────────────────────────

class TestNaiveRankingModel:
    def setup_method(self):
        self.model = NaiveRankingModel()

    def test_predict_returns_matchoutcome(self):
        result = self.model.predict(elo_a=1800, elo_b=1600)
        assert isinstance(result, MatchOutcome)

    def test_probabilities_sum_to_one(self):
        result = self.model.predict(elo_a=1800, elo_b=1600)
        total = result.p_win + result.p_draw + result.p_loss
        assert abs(total - 1.0) < 1e-6

    def test_better_team_has_higher_win_prob(self):
        result_favored = self.model.predict(elo_a=1900, elo_b=1500)
        result_underdog = self.model.predict(elo_a=1500, elo_b=1900)
        assert result_favored.p_win > result_underdog.p_win

    def test_equal_teams_near_even(self):
        result = self.model.predict(elo_a=1700, elo_b=1700)
        # Times iguais: win e loss devem ser aproximadamente iguais
        assert abs(result.p_win - result.p_loss) < 0.05

    def test_expected_goals_positive(self):
        result = self.model.predict(elo_a=1800, elo_b=1600)
        assert result.expected_goals_for > 0
        assert result.expected_goals_against > 0


# ── Testes PoissonModel ────────────────────────────────────────────────────────

class TestPoissonModel:
    def setup_method(self):
        self.model = PoissonModel()

    def test_predict_returns_matchoutcome(self):
        result = self.model.predict("Brasil", "Argentina")
        assert isinstance(result, MatchOutcome)

    def test_probabilities_sum_to_one(self):
        result = self.model.predict("Brasil", "Argentina")
        total = result.p_win + result.p_draw + result.p_loss
        assert abs(total - 1.0) < 1e-4

    def test_score_matrix_shape(self):
        result = self.model.predict("Brasil", "França")
        assert result.score_matrix is not None
        assert result.score_matrix.shape == (9, 9)

    def test_score_matrix_sums_to_one(self):
        result = self.model.predict("Alemanha", "Espanha")
        assert abs(result.score_matrix.sum() - 1.0) < 1e-4

    def test_strong_team_higher_xg(self):
        result_strong = self.model.predict("Argentina", "Iêmen")
        result_weak = self.model.predict("Iêmen", "Argentina")
        assert result_strong.expected_goals_for > result_weak.expected_goals_for

    def test_knockout_no_draw(self):
        result = self.model.predict_knockout("Brasil", "Argentina")
        assert result.p_draw == 0.0

    def test_knockout_probabilities_sum_to_one(self):
        result = self.model.predict_knockout("França", "Espanha")
        total = result.p_win + result.p_draw + result.p_loss
        assert abs(total - 1.0) < 1e-4

    def test_simulate_match_returns_valid_score(self):
        for _ in range(20):
            ga, gb = self.model.simulate_match("Brasil", "Alemanha")
            assert ga >= 0
            assert gb >= 0
            assert ga <= 8
            assert gb <= 8

    def test_unknown_team_uses_default(self):
        # Não deve lançar exceção com time desconhecido
        result = self.model.predict("Time Desconhecido X", "Brasil")
        total = result.p_win + result.p_draw + result.p_loss
        assert abs(total - 1.0) < 1e-4

    def test_most_likely_score_valid(self):
        result = self.model.predict("Brasil", "Argentina")
        score = result.most_likely_score()
        assert len(score) == 2
        assert score[0] >= 0
        assert score[1] >= 0


# ── Testes Features de Ranking ─────────────────────────────────────────────────

class TestRankingFeatures:
    def test_build_ranking_features_returns_dataframe(self):
        df = build_ranking_features(["Brasil", "Argentina"])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "fifa_position" in df.columns

    def test_all_required_columns_present(self):
        df = build_ranking_features(["França"])
        required = ["fifa_position", "fifa_points", "fifa_position_norm",
                    "fifa_points_norm", "tier", "elo_estimated"]
        for col in required:
            assert col in df.columns, f"Coluna ausente: {col}"

    def test_better_team_has_lower_position_number(self):
        df = build_ranking_features(["Argentina", "Iêmen"])
        assert df.loc["Argentina", "fifa_position"] < df.loc["Iêmen", "fifa_position"]

    def test_matchup_features_returns_dict(self):
        df = get_all_teams_ranking()
        features = compute_ranking_matchup_features("Brasil", "Argentina", df)
        assert isinstance(features, dict)
        assert "elo_win_probability" in features

    def test_elo_win_probability_range(self):
        df = get_all_teams_ranking()
        features = compute_ranking_matchup_features("Argentina", "Iêmen", df)
        assert 0 < features["elo_win_probability"] < 1

    def test_unknown_team_handled(self):
        df = build_ranking_features(["Time Inexistente"])
        assert len(df) == 1
        assert df.iloc[0]["fifa_position"] == 80  # valor default


# ── Testes Features de Forma ───────────────────────────────────────────────────

class TestFormFeatures:
    def test_returns_dict_with_keys(self, sample_results):
        features = compute_form_features(
            sample_results, "Brasil", "2025-01-01", n_games=10
        )
        required_keys = [
            "form_wins", "form_draws", "form_losses",
            "form_win_rate", "form_weighted_points",
        ]
        for key in required_keys:
            assert key in features, f"Chave ausente: {key}"

    def test_win_rate_between_0_and_1(self, sample_results):
        features = compute_form_features(
            sample_results, "Brasil", "2025-01-01"
        )
        assert 0 <= features["form_win_rate"] <= 1
        assert 0 <= features["form_draw_rate"] <= 1
        assert 0 <= features["form_loss_rate"] <= 1

    def test_rates_sum_to_one(self, sample_results):
        features = compute_form_features(
            sample_results, "Brasil", "2025-01-01"
        )
        total = (features["form_win_rate"]
                 + features["form_draw_rate"]
                 + features["form_loss_rate"])
        assert abs(total - 1.0) < 1e-6

    def test_no_data_returns_default(self, sample_results):
        # Time sem jogos antes da data
        features = compute_form_features(
            sample_results, "Brasil", "2020-01-01"
        )
        assert features["form_games_played"] == 0


# ── Testes Features H2H ────────────────────────────────────────────────────────

class TestH2HFeatures:
    def test_returns_dict(self, sample_results):
        features = compute_h2h_features(
            sample_results, "Brasil", "Argentina", "2025-01-01"
        )
        assert isinstance(features, dict)

    def test_no_h2h_returns_defaults(self, sample_results):
        features = compute_h2h_features(
            sample_results, "Brasil", "Catar", "2025-01-01"
        )
        assert features["h2h_total_games"] == 0
        assert features["h2h_win_rate_a"] == pytest.approx(0.33)

    def test_h2h_win_rates_sum_to_one(self, sample_results):
        features = compute_h2h_features(
            sample_results, "Brasil", "Argentina", "2025-01-01"
        )
        total = (features["h2h_win_rate_a"]
                 + features["h2h_draw_rate"]
                 + features["h2h_win_rate_b"])
        assert abs(total - 1.0) < 1e-6

    def test_dominance_range(self, sample_results):
        features = compute_h2h_features(
            sample_results, "Brasil", "Argentina", "2025-01-01"
        )
        assert -1.0 <= features["h2h_dominance"] <= 1.0
