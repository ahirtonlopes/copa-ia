"""
CopIA — Testes do Motor de Simulação Monte Carlo
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.baseline import PoissonModel
from src.simulation.monte_carlo import (
    TournamentSimulator,
    SimulationResults,
    COPA_2026_GROUPS,
)


@pytest.fixture
def simulator():
    return TournamentSimulator(model=PoissonModel(), random_seed=42)


class TestTournamentSimulator:
    def test_simulate_match_returns_valid_winner(self, simulator):
        winner, score = simulator.simulate_match("Brasil", "Argentina")
        assert winner in ["Brasil", "Argentina", "draw"]
        assert len(score) == 2
        assert score[0] >= 0 and score[1] >= 0

    def test_simulate_knockout_no_draw(self, simulator):
        for _ in range(20):
            winner, _ = simulator.simulate_match("Brasil", "Argentina", knockout=True)
            assert winner != "draw"

    def test_simulate_group_returns_4_teams(self, simulator):
        teams = ["Brasil", "Japão", "Nigéria", "Costa do Marfim"]
        standings = simulator.simulate_group(teams)
        assert len(standings) == 4
        team_names = [s["team"] for s in standings]
        for team in teams:
            assert team in team_names

    def test_simulate_group_points_valid(self, simulator):
        teams = ["Brasil", "Japão", "Nigéria", "Costa do Marfim"]
        standings = simulator.simulate_group(teams)
        for s in standings:
            assert 0 <= s["points"] <= 9  # Máximo 3 jogos × 3 pontos

    def test_simulate_group_ranks_from_1_to_4(self, simulator):
        teams = COPA_2026_GROUPS["D"]
        standings = simulator.simulate_group(teams)
        ranks = [s["rank"] for s in standings]
        assert sorted(ranks) == [1, 2, 3, 4]

    def test_simulate_all_groups_returns_all_groups(self, simulator):
        results = simulator.simulate_all_groups()
        assert len(results) == len(COPA_2026_GROUPS)
        for group in COPA_2026_GROUPS:
            assert group in results

    def test_get_qualified_teams(self, simulator):
        group_results = simulator.simulate_all_groups()
        first, second, thirds = simulator.get_qualified_teams(group_results)
        assert len(first) == len(COPA_2026_GROUPS)
        assert len(second) == len(COPA_2026_GROUPS)
        assert len(thirds) == 8

    def test_small_simulation_runs_without_error(self, simulator):
        results = simulator.run(n_simulations=100)
        assert isinstance(results, SimulationResults)
        assert results.n_simulations == 100

    def test_champion_probabilities_sum_to_approx_one(self, simulator):
        results = simulator.run(n_simulations=500)
        champ_probs = results.probabilities.get("champion", {})
        total = sum(champ_probs.values())
        # Pode ser ligeiramente diferente de 1 por times não classificados
        assert 0.95 <= total <= 1.05

    def test_champion_probabilities_positive(self, simulator):
        results = simulator.run(n_simulations=200)
        for prob in results.probabilities["champion"].values():
            assert 0 <= prob <= 1


class TestSimulationResults:
    def test_get_champion_probabilities_returns_df(self):
        mock_probs = {
            "champion": {"Brasil": 0.2, "Argentina": 0.18, "França": 0.15},
            "finalist": {"Brasil": 0.4, "Argentina": 0.36},
        }
        results = SimulationResults(mock_probs, n_simulations=1000)
        df = results.get_champion_probabilities()
        assert len(df) > 0
        assert "team" in df.columns
        assert "p_champion" in df.columns

    def test_get_full_probabilities_contains_all_teams(self):
        mock_probs = {
            "champion": {"Brasil": 0.2, "Argentina": 0.18},
            "finalist": {"Brasil": 0.4, "França": 0.3},
        }
        results = SimulationResults(mock_probs, n_simulations=1000)
        full_df = results.get_full_probabilities()
        team_names = full_df["team"].tolist()
        assert "Brasil" in team_names
        assert "França" in team_names
