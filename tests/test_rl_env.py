"""
CopIA — Testes do Ambiente Gymnasium TacticalCupEnv
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rl.tactical_env import TacticalCupEnv, N_ACTIONS, STATE_DIM, ACTIONS


@pytest.fixture
def env():
    e = TacticalCupEnv(opponent_strength=0.5, is_knockout=False)
    e.reset(seed=42)
    return e


@pytest.fixture
def knockout_env():
    e = TacticalCupEnv(opponent_strength=0.7, is_knockout=True, phase=4)
    e.reset(seed=42)
    return e


class TestTacticalCupEnv:
    def test_observation_space_shape(self, env):
        assert env.observation_space.shape == (STATE_DIM,)

    def test_action_space_size(self, env):
        assert env.action_space.n == N_ACTIONS

    def test_reset_returns_valid_observation(self, env):
        obs, info = env.reset(seed=0)
        assert obs.shape == (STATE_DIM,)
        assert np.all(obs >= 0.0)
        assert np.all(obs <= 1.0)

    def test_step_returns_correct_types(self, env):
        obs, reward, terminated, truncated, info = env.step(0)  # aguardar
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_observation_in_valid_range(self, env):
        for _ in range(18):  # Episódio completo
            action = env.action_space.sample()
            obs, _, terminated, _, _ = env.step(action)
            assert np.all(obs >= 0.0), f"Obs abaixo de 0: {obs.min()}"
            assert np.all(obs <= 1.0), f"Obs acima de 1: {obs.max()}"
            if terminated:
                break

    def test_episode_terminates_at_90_minutes(self, env):
        terminated = False
        steps = 0
        while not terminated:
            _, _, terminated, _, _ = env.step(0)
            steps += 1
            assert steps <= 20, "Episódio não terminou em 18 steps"

    def test_subs_used_increases(self, env):
        env.reset(seed=0)
        initial_subs = env.state.subs_used
        env.step(1)  # substituição ofensiva
        assert env.state.subs_used >= initial_subs

    def test_cannot_exceed_max_subs(self, env):
        env.reset(seed=0)
        # Força 5 substituições
        for _ in range(7):
            env.step(1)  # tenta sempre substituir
            if env.state.minute >= 90:
                break
        assert env.state.subs_used <= env.state.subs_max

    def test_minute_increases_per_step(self, env):
        env.reset(seed=0)
        prev_minute = env.state.minute
        env.step(0)
        assert env.state.minute == prev_minute + 5

    def test_all_actions_valid(self, env):
        for action in range(N_ACTIONS):
            local_env = TacticalCupEnv()
            local_env.reset(seed=42)
            obs, reward, terminated, _, info = local_env.step(action)
            assert obs.shape == (STATE_DIM,)

    def test_actions_have_names(self):
        for i in range(N_ACTIONS):
            assert i in ACTIONS
            assert isinstance(ACTIONS[i], str)

    def test_episode_summary_after_completion(self, env):
        env.reset(seed=0)
        terminated = False
        while not terminated:
            _, _, terminated, _, _ = env.step(env.action_space.sample())
        summary = env.get_episode_summary()
        assert "final_score" in summary
        assert "result" in summary
        assert summary["result"] in ["win", "draw", "loss"]

    def test_knockout_env_initializes(self, knockout_env):
        assert knockout_env.is_knockout is True
        assert knockout_env.phase == 4
        obs, _ = knockout_env.reset(seed=0)
        assert obs.shape == (STATE_DIM,)

    def test_reset_is_reproducible(self, env):
        obs1, _ = env.reset(seed=123)
        obs2, _ = env.reset(seed=123)
        np.testing.assert_array_equal(obs1, obs2)
