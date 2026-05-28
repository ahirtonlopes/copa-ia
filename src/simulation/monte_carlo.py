"""
CopIA — Motor de Simulação Monte Carlo
Simula o torneio completo da Copa do Mundo 2026 N vezes
e retorna distribuições de probabilidade por fase.
"""

import random
from collections import Counter, defaultdict
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from src.models.baseline import PoissonModel


# Grupos da Copa 2026
COPA_2026_GROUPS: dict[str, list[str]] = {
    "A": ["Estados Unidos", "Panamá", "Bolívia", "Nova Zelândia"],
    "B": ["Argentina", "Canadá", "Chile", "Peru"],
    "C": ["México", "Equador", "Venezuela", "Iraque"],
    "D": ["Brasil", "Japão", "Nigéria", "Costa do Marfim"],
    "E": ["Espanha", "Turquia", "Sérvia", "Argélia"],
    "F": ["França", "Bélgica", "Israel", "República Centro-Africana"],
    "G": ["Alemanha", "Eslovênia", "Hungria", "Arábia Saudita"],
    "H": ["Portugal", "Croácia", "Romênia", "Uruguai"],
    "I": ["Inglaterra", "Senegal", "Eslováquia", "Camarões"],
    "J": ["Países Baixos", "Dinamarca", "África do Sul", "Iêmen"],
    "K": ["Coreia do Sul", "Colômbia", "Costa Rica", "Gana"],
    "L": ["Marrocos", "Chile Alt", "Congo", "Zâmbia"],
}


class TournamentSimulator:
    """
    Simula o torneio completo da Copa do Mundo 2026.

    Formato 2026:
    - 48 times em 12 grupos de 4
    - Top 2 de cada grupo + 8 melhores terceiros avançam (32 times)
    - Fase eliminatória: oitavas → quartas → semifinais → final
    """

    def __init__(
        self,
        model: PoissonModel | None = None,
        ml_predict_fn: Any | None = None,
        random_seed: int = 42,
    ):
        """
        Args:
            model: Modelo Poisson (default se ml_predict_fn não fornecido)
            ml_predict_fn: Função de predição do ensemble ML (opcional)
            random_seed: Semente para reprodutibilidade
        """
        self.model = model or PoissonModel()
        self.ml_predict_fn = ml_predict_fn
        self.random_seed = random_seed

    def simulate_match(
        self,
        team_a: str,
        team_b: str,
        knockout: bool = False,
    ) -> tuple[str, tuple[int, int]]:
        """
        Simula um único jogo. Retorna (vencedor, (gols_a, gols_b)).
        Em mata-mata, prorrogação e pênaltis resolvem empates.
        """
        goals_a, goals_b = self.model.simulate_match(team_a, team_b, neutral=True)

        # Em mata-mata, empate vai para pênaltis (50/50 + ruído)
        if knockout and goals_a == goals_b:
            winner = team_a if random.random() < 0.50 else team_b
            return winner, (goals_a, goals_b)

        if goals_a > goals_b:
            return team_a, (goals_a, goals_b)
        elif goals_b > goals_a:
            return team_b, (goals_a, goals_b)
        else:
            return "draw", (goals_a, goals_b)

    def simulate_group(self, teams: list[str]) -> list[dict]:
        """
        Simula a fase de grupos (round-robin) para um grupo de 4 times.

        Returns:
            Lista de times classificados em ordem (1º ao 4º lugar)
            com pontos, saldo de gols e gols marcados.
        """
        standings = {
            team: {"points": 0, "gf": 0, "ga": 0, "gd": 0}
            for team in teams
        }

        # Gera todos os jogos do grupo
        for i, team_a in enumerate(teams):
            for team_b in teams[i + 1:]:
                winner, (ga, gb) = self.simulate_match(team_a, team_b, knockout=False)

                standings[team_a]["gf"] += ga
                standings[team_a]["ga"] += gb
                standings[team_b]["gf"] += gb
                standings[team_b]["ga"] += ga

                if winner == team_a:
                    standings[team_a]["points"] += 3
                elif winner == team_b:
                    standings[team_b]["points"] += 3
                else:  # empate
                    standings[team_a]["points"] += 1
                    standings[team_b]["points"] += 1

        # Calcula saldo de gols
        for team in standings:
            standings[team]["gd"] = standings[team]["gf"] - standings[team]["ga"]

        # Ordena: 1. pontos 2. saldo de gols 3. gols marcados 4. aleatório (critério de desempate)
        sorted_teams = sorted(
            standings.items(),
            key=lambda x: (x[1]["points"], x[1]["gd"], x[1]["gf"], random.random()),
            reverse=True,
        )

        return [
            {"team": team, "rank": rank + 1, **stats}
            for rank, (team, stats) in enumerate(sorted_teams)
        ]

    def simulate_all_groups(self) -> dict[str, list[dict]]:
        """Simula todos os 12 grupos."""
        return {
            group: self.simulate_group(teams)
            for group, teams in COPA_2026_GROUPS.items()
        }

    def get_qualified_teams(
        self, group_results: dict[str, list[dict]]
    ) -> tuple[dict[str, str], list[dict]]:
        """
        Determina os 32 times classificados para as oitavas.
        Top 2 de cada grupo (24 times) + 8 melhores terceiros.

        Returns:
            (qualificados_por_grupo, lista_dos_terceiros_classificados)
        """
        first_places = {}
        second_places = {}
        third_places = []

        for group, standings in group_results.items():
            first_places[group] = standings[0]["team"]
            second_places[group] = standings[1]["team"]
            third_places.append({
                "group": group,
                "team": standings[2]["team"],
                "points": standings[2]["points"],
                "gd": standings[2]["gd"],
                "gf": standings[2]["gf"],
            })

        # Os 8 melhores terceiros avançam
        best_thirds = sorted(
            third_places,
            key=lambda x: (x["points"], x["gd"], x["gf"], random.random()),
            reverse=True,
        )[:8]

        return first_places, second_places, best_thirds

    def simulate_knockout_bracket(
        self,
        first_places: dict[str, str],
        second_places: dict[str, str],
        best_thirds: list[dict],
    ) -> dict[str, Any]:
        """
        Simula o mata-mata completo (oitavas até final).

        Returns:
            Dict com o campeão e histórico de cada fase.
        """
        # Monta chaveamento das oitavas (simplificado — pode ser configurável)
        round_of_16_teams = (
            list(first_places.values())
            + list(second_places.values())
            + [t["team"] for t in best_thirds]
        )

        # Embaralha para criar confrontos das oitavas
        # (Na Copa real, o chaveamento é fixo — aqui simplificamos)
        random.shuffle(round_of_16_teams)
        matches = [
            (round_of_16_teams[i], round_of_16_teams[i + 1])
            for i in range(0, 32, 2)
        ]

        history: dict[str, list] = {
            "round_of_16": [],
            "quarterfinals": [],
            "semifinals": [],
            "final": [],
        }

        # Oitavas
        winners = []
        for team_a, team_b in matches:
            winner, score = self.simulate_match(team_a, team_b, knockout=True)
            history["round_of_16"].append({
                "team_a": team_a, "team_b": team_b,
                "winner": winner, "score": score,
            })
            winners.append(winner)

        # Quartas de final
        qf_matches = [(winners[i], winners[i + 1]) for i in range(0, 16, 2)]
        qf_winners = []
        for team_a, team_b in qf_matches:
            winner, score = self.simulate_match(team_a, team_b, knockout=True)
            history["quarterfinals"].append({
                "team_a": team_a, "team_b": team_b,
                "winner": winner, "score": score,
            })
            qf_winners.append(winner)

        # Semifinais
        sf_matches = [(qf_winners[i], qf_winners[i + 1]) for i in range(0, 8, 2)]
        sf_winners = []
        for team_a, team_b in sf_matches:
            winner, score = self.simulate_match(team_a, team_b, knockout=True)
            history["semifinals"].append({
                "team_a": team_a, "team_b": team_b,
                "winner": winner, "score": score,
            })
            sf_winners.append(winner)

        # Final
        champion_team, final_score = (
            sf_winners[0], sf_winners[1]
        ) if len(sf_winners) >= 2 else (sf_winners[0], sf_winners[0])

        # Simplificação: pega os 2 primeiros semifinalistas para a final
        finalists = sf_winners[:2]
        if len(finalists) == 2:
            champion, score = self.simulate_match(
                finalists[0], finalists[1], knockout=True
            )
        else:
            champion = finalists[0]
            score = (0, 0)

        history["final"].append({
            "team_a": finalists[0] if len(finalists) > 1 else "TBD",
            "team_b": finalists[1] if len(finalists) > 1 else "TBD",
            "winner": champion,
            "score": score,
        })

        return {"champion": champion, "history": history, "finalists": finalists}

    def run(self, n_simulations: int = 100_000) -> "SimulationResults":
        """
        Executa N simulações completas do torneio.

        Returns:
            SimulationResults com distribuições de probabilidade.
        """
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)

        counters: dict[str, Counter] = {
            "champion": Counter(),
            "finalist": Counter(),
            "semifinalist": Counter(),
            "quarterfinalist": Counter(),
            "round_of_16": Counter(),
            "group_winner": Counter(),
            "eliminated_groups": Counter(),
        }

        logger.info(f"Iniciando {n_simulations:,} simulações do torneio...")

        for sim in tqdm(range(n_simulations), desc="Simulando"):
            # 1. Fase de grupos
            group_results = self.simulate_all_groups()
            first, second, thirds = self.get_qualified_teams(group_results)

            # Registra quem saiu na fase de grupos
            for group, standings in group_results.items():
                counters["group_winner"][standings[0]["team"]] += 1
                for standing in standings[2:]:  # 3º e 4º eliminados
                    counters["eliminated_groups"][standing["team"]] += 1

            # 2. Mata-mata
            knockout_result = self.simulate_knockout_bracket(first, second, thirds)

            champion = knockout_result["champion"]
            finalists = knockout_result["finalists"]
            history = knockout_result["history"]

            counters["champion"][champion] += 1

            for team in finalists:
                counters["finalist"][team] += 1

            for match in history.get("semifinals", []):
                counters["semifinalist"][match["team_a"]] += 1
                counters["semifinalist"][match["team_b"]] += 1

            for match in history.get("quarterfinals", []):
                counters["quarterfinalist"][match["team_a"]] += 1
                counters["quarterfinalist"][match["team_b"]] += 1

            for match in history.get("round_of_16", []):
                counters["round_of_16"][match["team_a"]] += 1
                counters["round_of_16"][match["team_b"]] += 1

        # Normaliza para probabilidades
        probabilities = {
            stage: {
                team: count / n_simulations
                for team, count in counter.most_common()
            }
            for stage, counter in counters.items()
        }

        logger.success(f"Simulação concluída! Campeão mais provável: "
                       f"{list(probabilities['champion'].items())[0]}")

        return SimulationResults(
            probabilities=probabilities,
            n_simulations=n_simulations,
        )


class SimulationResults:
    """Container para resultados de simulação com métodos de análise e exportação."""

    def __init__(
        self,
        probabilities: dict[str, dict[str, float]],
        n_simulations: int,
    ):
        self.probabilities = probabilities
        self.n_simulations = n_simulations

    def get_champion_probabilities(self, top_n: int = 20) -> pd.DataFrame:
        """Retorna DataFrame com probabilidades de título."""
        data = list(self.probabilities.get("champion", {}).items())
        data.sort(key=lambda x: x[1], reverse=True)
        df = pd.DataFrame(data[:top_n], columns=["team", "p_champion"])
        df["p_champion_pct"] = (df["p_champion"] * 100).round(1)
        return df

    def get_full_probabilities(self) -> pd.DataFrame:
        """Retorna DataFrame consolidado com probabilidades por fase."""
        all_teams = set()
        for stage_probs in self.probabilities.values():
            all_teams.update(stage_probs.keys())

        rows = []
        for team in sorted(all_teams):
            row = {"team": team}
            for stage, probs in self.probabilities.items():
                row[f"p_{stage}"] = probs.get(team, 0.0)
            rows.append(row)

        df = pd.DataFrame(rows)
        df = df.sort_values("p_champion", ascending=False).reset_index(drop=True)
        return df

    def print_summary(self, top_n: int = 10) -> None:
        """Imprime sumário das simulações."""
        champ_df = self.get_champion_probabilities(top_n)
        print(f"\n{'='*50}")
        print(f"Copa do Mundo 2026 — {self.n_simulations:,} simulações")
        print(f"{'='*50}")
        print(f"\nTop {top_n} candidatos ao título:")
        for _, row in champ_df.iterrows():
            bar = "█" * int(row["p_champion_pct"] * 2)
            print(f"  {row['team']:<25} {row['p_champion_pct']:5.1f}% {bar}")

    def to_dict(self) -> dict:
        return {
            "probabilities": self.probabilities,
            "n_simulations": self.n_simulations,
        }
