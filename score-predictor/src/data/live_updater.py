"""
CopIA Score Predictor — Live Updater
Adiciona resultados reais da Copa 2026 conforme os jogos acontecem.

Uso rápido:
    from src.data.live_updater import LiveUpdater
    updater = LiveUpdater()
    updater.add_result("Brasil", "Nigéria", 2, 0, "2026-06-15", "FIFA World Cup")
    updater.save()
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from loguru import logger

ROOT = Path(__file__).parent.parent.parent
LIVE_DIR = ROOT / "data" / "live"
LIVE_FILE = LIVE_DIR / "copa2026_results.json"
LIVE_CSV = LIVE_DIR / "copa2026_results.csv"


# ── Jogos da Copa 2026 (schedule completo da fase de grupos) ───────────────
# Formato: (home_team, away_team, date, venue, group)
COPA_2026_SCHEDULE: list[dict] = [
    # Grupo A — EUA, Panamá, Bolívia, Nova Zelândia
    {"home": "Estados Unidos", "away": "Panamá",       "date": "2026-06-11", "group": "A"},
    {"home": "Bolívia",        "away": "Nova Zelândia", "date": "2026-06-11", "group": "A"},
    {"home": "Estados Unidos", "away": "Bolívia",       "date": "2026-06-15", "group": "A"},
    {"home": "Panamá",         "away": "Nova Zelândia", "date": "2026-06-15", "group": "A"},
    {"home": "Estados Unidos", "away": "Nova Zelândia", "date": "2026-06-19", "group": "A"},
    {"home": "Panamá",         "away": "Bolívia",       "date": "2026-06-19", "group": "A"},
    # Grupo B — Argentina, Canadá, Chile, Peru
    {"home": "Argentina",      "away": "Canadá",        "date": "2026-06-12", "group": "B"},
    {"home": "Chile",          "away": "Peru",           "date": "2026-06-12", "group": "B"},
    {"home": "Argentina",      "away": "Chile",          "date": "2026-06-16", "group": "B"},
    {"home": "Canadá",         "away": "Peru",           "date": "2026-06-16", "group": "B"},
    {"home": "Argentina",      "away": "Peru",           "date": "2026-06-20", "group": "B"},
    {"home": "Canadá",         "away": "Chile",          "date": "2026-06-20", "group": "B"},
    # Grupo D — Brasil, Japão, Nigéria, Costa do Marfim
    {"home": "Brasil",         "away": "Costa do Marfim","date": "2026-06-13", "group": "D"},
    {"home": "Japão",          "away": "Nigéria",        "date": "2026-06-13", "group": "D"},
    {"home": "Brasil",         "away": "Nigéria",        "date": "2026-06-17", "group": "D"},
    {"home": "Costa do Marfim","away": "Japão",          "date": "2026-06-17", "group": "D"},
    {"home": "Brasil",         "away": "Japão",          "date": "2026-06-21", "group": "D"},
    {"home": "Nigéria",        "away": "Costa do Marfim","date": "2026-06-21", "group": "D"},
    # Grupo E — Espanha, Turquia, Sérvia, Argélia
    {"home": "Espanha",        "away": "Sérvia",         "date": "2026-06-13", "group": "E"},
    {"home": "Turquia",        "away": "Argélia",        "date": "2026-06-13", "group": "E"},
    {"home": "Espanha",        "away": "Turquia",        "date": "2026-06-17", "group": "E"},
    {"home": "Sérvia",         "away": "Argélia",        "date": "2026-06-17", "group": "E"},
    {"home": "Espanha",        "away": "Argélia",        "date": "2026-06-21", "group": "E"},
    {"home": "Turquia",        "away": "Sérvia",         "date": "2026-06-21", "group": "E"},
    # Grupo F — França, Bélgica, Israel, Rep. Centro-Africana
    {"home": "França",         "away": "Bélgica",        "date": "2026-06-14", "group": "F"},
    {"home": "França",         "away": "Israel",         "date": "2026-06-18", "group": "F"},
    {"home": "Bélgica",        "away": "Israel",         "date": "2026-06-22", "group": "F"},
    # Grupo H — Portugal, Croácia, Romênia, Uruguai
    {"home": "Portugal",       "away": "Croácia",        "date": "2026-06-14", "group": "H"},
    {"home": "Romênia",        "away": "Uruguai",        "date": "2026-06-14", "group": "H"},
    {"home": "Portugal",       "away": "Uruguai",        "date": "2026-06-18", "group": "H"},
    {"home": "Croácia",        "away": "Romênia",        "date": "2026-06-18", "group": "H"},
    {"home": "Portugal",       "away": "Romênia",        "date": "2026-06-22", "group": "H"},
    {"home": "Croácia",        "away": "Uruguai",        "date": "2026-06-22", "group": "H"},
    # Grupo I — Inglaterra, Senegal, Eslováquia, Camarões
    {"home": "Inglaterra",     "away": "Senegal",        "date": "2026-06-15", "group": "I"},
    {"home": "Eslováquia",     "away": "Camarões",       "date": "2026-06-15", "group": "I"},
    {"home": "Inglaterra",     "away": "Eslováquia",     "date": "2026-06-19", "group": "I"},
    {"home": "Senegal",        "away": "Camarões",       "date": "2026-06-19", "group": "I"},
    {"home": "Inglaterra",     "away": "Camarões",       "date": "2026-06-23", "group": "I"},
    {"home": "Senegal",        "away": "Eslováquia",     "date": "2026-06-23", "group": "I"},
]


class LiveUpdater:
    """
    Gerencia os resultados reais da Copa 2026 em tempo real.

    Uso:
        updater = LiveUpdater()

        # Adicionar resultado após o jogo
        updater.add_result("Brasil", "Nigéria", 2, 0, "2026-06-17")

        # Ver o que já foi jogado
        updater.show_results()

        # Exportar para o pipeline de modelos
        df = updater.to_dataframe()
    """

    def __init__(self):
        LIVE_DIR.mkdir(parents=True, exist_ok=True)
        self.results: list[dict] = []
        self._load()

    def _load(self):
        """Carrega resultados já registrados."""
        if LIVE_FILE.exists():
            with open(LIVE_FILE) as f:
                self.results = json.load(f)
            logger.info(f"Carregados {len(self.results)} resultados da Copa 2026")

    def add_result(
        self,
        home_team: str,
        away_team: str,
        home_score: int,
        away_score: int,
        date: str,
        tournament: str = "FIFA World Cup",
        neutral: bool = True,
        notes: str = "",
    ) -> None:
        """
        Registra o resultado de um jogo.

        Args:
            home_team: Nome da seleção da casa (ou primeiro time em campo neutro)
            away_team: Nome da seleção visitante
            home_score: Gols do time da casa (tempo regulamentar)
            away_score: Gols do time visitante (tempo regulamentar)
            date: Data do jogo (YYYY-MM-DD)
            tournament: Nome da competição
            neutral: True para campo neutro (todos os jogos da Copa)
            notes: Observações opcionais (ex: "pênaltis", "prorrogação")
        """
        # Verifica se já existe
        for r in self.results:
            if (r["home_team"] == home_team
                    and r["away_team"] == away_team
                    and r["date"] == date):
                logger.warning(f"Resultado já registrado: {home_team} x {away_team} ({date}). Atualizando...")
                r.update({
                    "home_score": home_score,
                    "away_score": away_score,
                    "notes": notes,
                    "updated_at": datetime.now().isoformat(),
                })
                self.save()
                return

        result = {
            "date": date,
            "home_team": home_team,
            "away_team": away_team,
            "home_score": home_score,
            "away_score": away_score,
            "tournament": tournament,
            "neutral": neutral,
            "notes": notes,
            "added_at": datetime.now().isoformat(),
        }
        self.results.append(result)
        self.save()

        diff = home_score - away_score
        outcome = "✓ Vitória" if diff > 0 else ("= Empate" if diff == 0 else "✗ Derrota")
        logger.success(
            f"{outcome}: {home_team} {home_score}–{away_score} {away_team} ({date})"
        )

    def show_results(self) -> None:
        """Imprime tabela de resultados registrados."""
        if not self.results:
            print("Nenhum resultado registrado ainda.")
            return

        print(f"\n{'='*60}")
        print(f"  Copa 2026 — {len(self.results)} resultado(s) registrado(s)")
        print(f"{'='*60}")
        for r in sorted(self.results, key=lambda x: x["date"]):
            print(
                f"  {r['date']}  "
                f"{r['home_team']:<22} "
                f"{r['home_score']}–{r['away_score']}  "
                f"{r['away_team']}"
                + (f"  [{r['notes']}]" if r.get("notes") else "")
            )
        print(f"{'='*60}\n")

    def to_dataframe(self) -> pd.DataFrame:
        """Converte resultados para DataFrame no formato padrão do pipeline."""
        if not self.results:
            return pd.DataFrame(columns=[
                "date", "home_team", "away_team",
                "home_score", "away_score", "tournament", "neutral",
            ])
        df = pd.DataFrame(self.results)
        df["date"] = pd.to_datetime(df["date"])
        return df[["date", "home_team", "away_team",
                   "home_score", "away_score", "tournament", "neutral"]]

    def pending_matches(self, as_of: str | None = None) -> list[dict]:
        """Lista jogos do schedule que ainda não têm resultado."""
        played = {
            (r["home_team"], r["away_team"], r["date"])
            for r in self.results
        }
        cutoff = as_of or datetime.today().strftime("%Y-%m-%d")
        return [
            g for g in COPA_2026_SCHEDULE
            if (g["home"], g["away"], g["date"]) not in played
            and g["date"] >= cutoff
        ]

    def save(self) -> None:
        """Persiste resultados em JSON e CSV."""
        with open(LIVE_FILE, "w") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

        df = self.to_dataframe()
        if not df.empty:
            df.to_csv(LIVE_CSV, index=False)

        logger.debug(f"Salvo: {len(self.results)} resultados em {LIVE_FILE}")
