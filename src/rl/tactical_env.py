"""
CopIA Tactics — Ambiente Gymnasium para Decisões Táticas
Fase 2 do projeto: treinamento de agente RL para decisões de substituição.

O treinador (agente) decide a cada 5 minutos:
- Fazer substituição ofensiva, defensiva ou de recuperação
- Mudar formação
- Aguardar

Objetivo: maximizar probabilidade de vitória ao final do jogo.
"""

from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass
class MatchState:
    """Estado completo de uma partida em um dado momento."""
    minute: int = 0
    score_own: int = 0
    score_opp: int = 0
    subs_used: int = 0
    subs_max: int = 5
    xg_for: float = 0.0
    xg_against: float = 0.0
    fatigue_attack: float = 0.0     # 0=fresco, 1=exausto
    fatigue_midfield: float = 0.0
    fatigue_defense: float = 0.0
    possession: float = 0.5         # 0-1
    pressure: float = 0.5           # pressão adversária 0-1
    yellow_cards_own: int = 0
    yellow_cards_opp: int = 0
    opponent_strength: float = 0.5  # 0=fraco, 1=muito forte
    is_knockout: bool = False
    phase: int = 0                  # 0=grupos, 1=oitavas, ..., 4=final
    last_sub_minute: int = -1
    formation_mode: int = 1         # 0=defensivo, 1=equilibrado, 2=ofensivo
    bench_quality: float = 0.5      # qualidade do banco 0-1


# Mapeamento das ações
ACTIONS = {
    0: "aguardar",
    1: "substituicao_ofensiva",
    2: "substituicao_defensiva",
    3: "substituicao_recuperacao",
    4: "formacao_ofensiva",
    5: "formacao_defensiva",
    6: "substituicao_dupla",
}

N_ACTIONS = len(ACTIONS)
STATE_DIM = 22


class TacticalCupEnv(gym.Env):
    """
    Ambiente Gymnasium para decisões táticas de futebol.

    Episódio:
    - Cada step = 5 minutos de jogo
    - 18 steps por episódio (90 minutos)
    - Termina quando minute >= 90

    Compatível com Stable-Baselines3 e CleanRL.
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        opponent_strength: float = 0.5,
        is_knockout: bool = False,
        phase: int = 0,
        render_mode: str | None = None,
    ):
        super().__init__()

        self.opponent_strength = opponent_strength
        self.is_knockout = is_knockout
        self.phase = phase
        self.render_mode = render_mode

        # Espaços Gymnasium
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(STATE_DIM,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(N_ACTIONS)

        # Estado interno
        self.state: MatchState = MatchState()
        self.history: list[dict] = []
        self.step_count: int = 0

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        # Inicializa com variabilidade aleatória para generalização
        self.state = MatchState(
            minute=0,
            score_own=0,
            score_opp=0,
            subs_used=0,
            subs_max=5,
            fatigue_attack=self.np_random.uniform(0.0, 0.1),
            fatigue_midfield=self.np_random.uniform(0.0, 0.1),
            fatigue_defense=self.np_random.uniform(0.0, 0.1),
            possession=self.np_random.uniform(0.4, 0.6),
            pressure=self.opponent_strength * 0.6 + self.np_random.uniform(0.0, 0.2),
            opponent_strength=self.opponent_strength,
            is_knockout=self.is_knockout,
            phase=self.phase,
            bench_quality=self.np_random.uniform(0.3, 0.8),
            formation_mode=1,
        )
        self.history = []
        self.step_count = 0

        return self._get_observation(), {}

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Executa uma ação (decisão do treinador a cada 5 minutos).

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        prev_state = self._copy_state()
        self._apply_action(action)
        self._simulate_5_minutes()

        reward = self._compute_reward(prev_state, action)
        self.state.minute += 5
        self.step_count += 1

        terminated = self.state.minute >= 90
        truncated = False

        # Recompensa terminal
        if terminated:
            reward += self._terminal_reward()

        info = {
            "minute": self.state.minute,
            "score": f"{self.state.score_own}-{self.state.score_opp}",
            "action": ACTIONS[action],
            "subs_remaining": self.state.subs_max - self.state.subs_used,
        }

        self.history.append({
            "minute": prev_state.minute,
            "action": ACTIONS[action],
            "score_before": f"{prev_state.score_own}-{prev_state.score_opp}",
            "score_after": f"{self.state.score_own}-{self.state.score_opp}",
            "reward": reward,
        })

        obs = self._get_observation()
        return obs, reward, terminated, truncated, info

    def _apply_action(self, action: int) -> None:
        """Aplica o efeito imediato da ação no estado."""
        s = self.state

        if action == 0:  # aguardar
            pass

        elif action == 1 and s.subs_used < s.subs_max:  # substituição ofensiva
            s.subs_used += 1
            s.last_sub_minute = s.minute
            # Efeito: reduz cansaço do ataque, aumenta pressão ofensiva
            s.fatigue_attack = max(0, s.fatigue_attack - 0.25 * s.bench_quality)
            s.possession = min(1.0, s.possession + 0.05)
            if s.formation_mode < 2:
                s.formation_mode = min(2, s.formation_mode + 0.3)

        elif action == 2 and s.subs_used < s.subs_max:  # substituição defensiva
            s.subs_used += 1
            s.last_sub_minute = s.minute
            s.fatigue_defense = max(0, s.fatigue_defense - 0.25 * s.bench_quality)
            s.pressure = max(0, s.pressure - 0.08)
            if s.formation_mode > 0:
                s.formation_mode = max(0, s.formation_mode - 0.3)

        elif action == 3 and s.subs_used < s.subs_max:  # substituição recuperação
            s.subs_used += 1
            s.last_sub_minute = s.minute
            s.fatigue_midfield = max(0, s.fatigue_midfield - 0.25 * s.bench_quality)

        elif action == 4:  # formação ofensiva
            s.formation_mode = 2
            s.possession = min(1.0, s.possession + 0.03)
            s.pressure = min(1.0, s.pressure + 0.05)  # adversário também pressiona mais

        elif action == 5:  # formação defensiva
            s.formation_mode = 0
            s.pressure = max(0, s.pressure - 0.05)
            s.possession = max(0, s.possession - 0.05)

        elif action == 6 and s.subs_used + 2 <= s.subs_max:  # substituição dupla
            s.subs_used += 2
            s.last_sub_minute = s.minute
            s.fatigue_attack = max(0, s.fatigue_attack - 0.20 * s.bench_quality)
            s.fatigue_midfield = max(0, s.fatigue_midfield - 0.15 * s.bench_quality)

    def _simulate_5_minutes(self) -> None:
        """Simula os efeitos naturais de 5 minutos de jogo no estado."""
        s = self.state
        rng = self.np_random

        # Cansaço aumenta com o tempo
        fatigue_increase = 0.04 + rng.uniform(0, 0.02)
        s.fatigue_attack = min(1.0, s.fatigue_attack + fatigue_increase)
        s.fatigue_midfield = min(1.0, s.fatigue_midfield + fatigue_increase * 0.9)
        s.fatigue_defense = min(1.0, s.fatigue_defense + fatigue_increase * 0.8)

        # xG acumulado (baseado em posse, formação e cansaço)
        xg_rate_for = (
            0.08                              # base
            + s.possession * 0.05             # posse ajuda
            + (s.formation_mode / 2) * 0.03  # modo ofensivo gera mais xG
            - s.fatigue_attack * 0.02         # cansaço reduz criatividade
        ) * rng.uniform(0.5, 1.5)            # variabilidade aleatória

        xg_rate_against = (
            0.07                               # base adversário
            + s.opponent_strength * 0.06       # adversário forte cria mais
            + s.pressure * 0.04                # pressão adversária
            - (1 - s.formation_mode / 2) * 0.02  # modo defensivo protege mais
        ) * rng.uniform(0.5, 1.5)

        s.xg_for += xg_rate_for
        s.xg_against += xg_rate_against

        # Gols: baseado em xG acumulado com threshold estocástico
        if rng.random() < xg_rate_for * 0.7:
            s.score_own += 1

        if rng.random() < xg_rate_against * 0.7:
            s.score_opp += 1

        # Cartões (raramente)
        if rng.random() < 0.02:
            s.yellow_cards_own += 1
        if rng.random() < 0.025 * s.opponent_strength:
            s.yellow_cards_opp += 1

        # Posse oscila naturalmente
        s.possession = np.clip(
            s.possession + rng.uniform(-0.03, 0.03), 0.3, 0.7
        )

    def _compute_reward(self, prev_state: MatchState, action: int) -> float:
        """
        Recompensa densa (shaping) para guiar o aprendizado.
        Reward shaping: xG delta, uso eficaz de substituições.
        """
        s = self.state
        reward = 0.0

        # Delta de xG após a ação (janela de 5 minutos)
        xg_for_delta = s.xg_for - prev_state.xg_for
        xg_against_delta = s.xg_against - prev_state.xg_against
        reward += 0.4 * xg_for_delta - 0.3 * xg_against_delta

        # Penalidade por desperdiçar substituição sem efeito
        if action in [1, 2, 3, 6]:
            sub_effect = (
                prev_state.fatigue_attack - s.fatigue_attack
                + prev_state.fatigue_midfield - s.fatigue_midfield
                + prev_state.fatigue_defense - s.fatigue_defense
            )
            if sub_effect < 0.05:
                reward -= 0.3  # Substituição sem efeito positivo

        # Incentivo por gol marcado
        if s.score_own > prev_state.score_own:
            reward += 2.0

        # Penalidade por gol sofrido
        if s.score_opp > prev_state.score_opp:
            reward -= 1.5

        return float(reward)

    def _terminal_reward(self) -> float:
        """Recompensa ao final dos 90 minutos."""
        s = self.state
        diff = s.score_own - s.score_opp

        if diff > 0:
            return 10.0 + min(diff - 1, 3) * 1.0  # Vitória + bônus por placar
        elif diff == 0:
            if s.is_knockout:
                return -1.0  # Empate em mata-mata (vai para pênaltis — incerto)
            return 3.0  # Empate na fase de grupos vale alguma coisa
        else:
            return -5.0 + max(diff + 1, -3) * 0.5  # Derrota

    def _get_observation(self) -> np.ndarray:
        """Converte o estado interno em vetor de observação normalizado."""
        s = self.state
        obs = np.array([
            s.minute / 90.0,
            min(s.score_own, 5) / 5.0,
            min(s.score_opp, 5) / 5.0,
            (s.score_own - s.score_opp + 5) / 10.0,  # saldo normalizado
            s.subs_used / s.subs_max,
            (s.subs_max - s.subs_used) / s.subs_max,
            s.fatigue_attack,
            s.fatigue_midfield,
            s.fatigue_defense,
            min(s.xg_for, 5) / 5.0,
            min(s.xg_against, 5) / 5.0,
            np.clip((s.xg_for - s.xg_against + 2.5) / 5.0, 0.0, 1.0),
            s.possession,
            s.pressure,
            min(s.yellow_cards_own, 5) / 5.0,
            min(s.yellow_cards_opp, 5) / 5.0,
            s.opponent_strength,
            s.bench_quality,
            float(s.is_knockout),
            s.phase / 4.0,
            (s.last_sub_minute + 1) / 91.0 if s.last_sub_minute >= 0 else 0.0,
            s.formation_mode / 2.0,
        ], dtype=np.float32)

        return np.clip(obs, 0.0, 1.0)

    def _copy_state(self) -> MatchState:
        """Cria cópia do estado atual."""
        import copy
        return copy.copy(self.state)

    def render(self) -> str | None:
        if self.render_mode == "ansi":
            s = self.state
            status = (
                f"Min {s.minute:2d} | "
                f"Placar: {s.score_own}-{s.score_opp} | "
                f"Subs: {s.subs_used}/{s.subs_max} | "
                f"xG: {s.xg_for:.2f}-{s.xg_against:.2f} | "
                f"Cansaço: ATK={s.fatigue_attack:.2f} MID={s.fatigue_midfield:.2f}"
            )
            print(status)
            return status
        return None

    def get_episode_summary(self) -> dict:
        """Retorna resumo do episódio para análise."""
        s = self.state
        return {
            "final_score": f"{s.score_own}-{s.score_opp}",
            "result": (
                "win" if s.score_own > s.score_opp
                else "draw" if s.score_own == s.score_opp
                else "loss"
            ),
            "total_reward": sum(h["reward"] for h in self.history),
            "subs_used": s.subs_used,
            "decisions": self.history,
            "xg_for": s.xg_for,
            "xg_against": s.xg_against,
        }
