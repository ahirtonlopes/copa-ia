"""
CopIA Score Predictor — Gerador de Relatórios
Produz tabelas, resumos e posts prontos para LinkedIn a partir das predições.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
from loguru import logger

ROOT = Path(__file__).parent.parent.parent
REPORTS_DIR = ROOT / "outputs" / "reports"


class ReportGenerator:
    """Gera relatórios e posts de mídia social a partir das predições."""

    def __init__(self, predictions: pd.DataFrame):
        """
        Args:
            predictions: DataFrame retornado por DixonColesModel.predict_batch()
                         com colunas: home, away, p_home_win, p_draw, p_away_win,
                         xg_home, xg_away, most_likely_score, confidence
        """
        self.preds = predictions
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    def match_card(self, home: str, away: str) -> str:
        """Card textual de uma partida específica."""
        row = self.preds[
            (self.preds["home"] == home) & (self.preds["away"] == away)
        ]
        if row.empty:
            return f"Predição não encontrada: {home} vs {away}"

        r = row.iloc[0]
        fav = home if r["p_home_win"] > r["p_away_win"] else away
        if r["p_draw"] > max(r["p_home_win"], r["p_away_win"]):
            fav = "Empate"

        return f"""
┌─────────────────────────────────────────────────┐
│  {home:<22} vs  {away:<22}│
├─────────────────────────────────────────────────┤
│  Placar mais provável : {r['most_likely_score']:<6}                    │
│  xG esperado          : {r['xg_home']:.1f} – {r['xg_away']:.1f}                   │
├─────────────────────────────────────────────────┤
│  Vitória {home[:10]:<10}  : {r['p_home_win']:>6.1%}                  │
│  Empate              : {r['p_draw']:>6.1%}                  │
│  Vitória {away[:10]:<10}  : {r['p_away_win']:>6.1%}                  │
├─────────────────────────────────────────────────┤
│  Favorito: {fav:<20} [{r['confidence']:<16}]│
└─────────────────────────────────────────────────┘"""

    def group_stage_table(self, group_name: str, teams: list[str]) -> str:
        """Tabela de probabilidades para uma fase de grupos."""
        group_preds = self.preds[
            self.preds["home"].isin(teams) | self.preds["away"].isin(teams)
        ]

        lines = [f"\n  GRUPO {group_name}", "  " + "─" * 60]
        lines.append(f"  {'Jogo':<40} {'Placar':>6}  {'Casa':>6}  {'Emp':>5}  {'Fora':>5}")
        lines.append("  " + "─" * 60)

        for _, r in group_preds.iterrows():
            match = f"{r['home']} × {r['away']}"
            lines.append(
                f"  {match:<40} {r['most_likely_score']:>6}  "
                f"{r['p_home_win']:>5.1%}  {r['p_draw']:>4.1%}  {r['p_away_win']:>4.1%}"
            )
        lines.append("  " + "─" * 60)
        return "\n".join(lines)

    def full_predictions_table(self) -> str:
        """Tabela completa de predições de todos os jogos."""
        lines = [
            "\n" + "="*70,
            f"  CopIA Score Predictor — Predições Copa 2026",
            f"  Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M')}",
            "="*70,
            f"  {'Jogo':<38} {'Placar':>6}  {'Casa':>6}  {'Emp':>5}  {'Fora':>5}  {'Conf.':>16}",
            "  " + "─"*68,
        ]
        for _, r in self.preds.iterrows():
            match = f"{r['home']} × {r['away']}"
            lines.append(
                f"  {match:<38} {r['most_likely_score']:>6}  "
                f"{r['p_home_win']:>5.1%}  {r['p_draw']:>4.1%}  "
                f"{r['p_away_win']:>4.1%}  {r['confidence']:>16}"
            )
        lines.append("="*70)
        return "\n".join(lines)

    def linkedin_post(
        self,
        focus_team: str = "Brasil",
        hashtags: bool = True,
    ) -> str:
        """
        Gera um post pronto para LinkedIn com as predições do time foco.
        """
        team_games = self.preds[
            (self.preds["home"] == focus_team) | (self.preds["away"] == focus_team)
        ]

        lines = [f"🤖 Copa 2026 — Predições do CopIA para o {focus_team}", ""]

        for _, r in team_games.iterrows():
            is_home = r["home"] == focus_team
            opp = r["away"] if is_home else r["home"]
            p_win = r["p_home_win"] if is_home else r["p_away_win"]
            p_lose = r["p_away_win"] if is_home else r["p_home_win"]

            score = r["most_likely_score"]
            if not is_home:
                parts = score.split("–")
                score = f"{parts[1]}–{parts[0]}" if len(parts) == 2 else score

            fav_emoji = "⚡" if p_win > 0.5 else ("⚖️" if r["p_draw"] > 0.35 else "⚠️")

            lines.append(
                f"{fav_emoji} {focus_team} vs {opp}: "
                f"Placar {score} | "
                f"Vitória {p_win:.0%} · Empate {r['p_draw']:.0%} · Derrota {p_lose:.0%}"
            )

        lines += [
            "",
            "📊 Modelo: Dixon-Coles ajustado em 40k+ jogos históricos",
            "🔁 Atualizado após cada rodada",
            "🔗 github.com/ahirtonlopes/copa-ia",
        ]

        if hashtags:
            lines += [
                "",
                "#CopIA #Copa2026 #MachineLearning #DataScience #Futebol "
                "#IA #Python #SportAnalytics",
            ]

        return "\n".join(lines)

    def save_all(self, prefix: str = "copa2026") -> dict[str, Path]:
        """Salva todos os relatórios em outputs/reports/."""
        saved = {}

        # CSV com predições
        csv_path = REPORTS_DIR / f"{prefix}_predictions.csv"
        self.preds.to_csv(csv_path, index=False)
        saved["csv"] = csv_path

        # Relatório textual
        txt_path = REPORTS_DIR / f"{prefix}_report.txt"
        txt_path.write_text(self.full_predictions_table(), encoding="utf-8")
        saved["txt"] = txt_path

        # Post LinkedIn
        post_path = REPORTS_DIR / f"{prefix}_linkedin_post.txt"
        post_path.write_text(self.linkedin_post(), encoding="utf-8")
        saved["linkedin"] = post_path

        logger.success(f"Relatórios salvos em {REPORTS_DIR}")
        for name, path in saved.items():
            logger.info(f"  {name}: {path.name}")

        return saved
