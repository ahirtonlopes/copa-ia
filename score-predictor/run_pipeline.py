"""
CopIA Score Predictor — Pipeline Principal
Executa o pipeline completo: download → limpeza → Elo → Dixon-Coles → predições → relatório.

Uso:
    uv run python run_pipeline.py                   # Pipeline completo
    uv run python run_pipeline.py --model elo       # Apenas Elo
    uv run python run_pipeline.py --model dc        # Apenas Dixon-Coles
    uv run python run_pipeline.py --skip-download   # Usa cache local
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlflow
import pandas as pd
from loguru import logger

# Garante que src/ está no path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.downloader import download_all
from src.data.live_updater import COPA_2026_SCHEDULE, LiveUpdater
from src.data.preprocessor import clean_results, save_processed
from src.models.dixon_coles import DixonColesModel
from src.models.elo import EloRatingSystem
from src.reporting.generator import ReportGenerator

MLFLOW_URI = str(Path(__file__).parent / "mlruns")


def parse_args():
    p = argparse.ArgumentParser(description="CopIA Score Predictor Pipeline")
    p.add_argument("--skip-download", action="store_true",
                   help="Pula download e usa cache em data/raw/")
    p.add_argument("--model", choices=["elo", "dc", "all"], default="all",
                   help="Qual modelo treinar (default: all)")
    p.add_argument("--n-simulations", type=int, default=50_000,
                   help="Simulações Monte Carlo (default: 50.000)")
    p.add_argument("--min-year", type=int, default=1990,
                   help="Ano mínimo dos dados para treino (default: 1990)")
    return p.parse_args()


def run(args=None):
    if args is None:
        args = parse_args()

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("copa-ia-score-predictor")

    with mlflow.start_run(run_name=f"pipeline_{pd.Timestamp.today().strftime('%Y%m%d_%H%M')}"):

        # ── 1. DADOS ──────────────────────────────────────────────────────
        logger.info("📥 ETAPA 1/5 — Dados históricos")
        if not args.skip_download:
            raw = download_all()
            df_raw = raw["results"]
        else:
            raw_path = Path(__file__).parent / "data" / "raw" / "results.csv"
            if not raw_path.exists():
                logger.error("Cache não encontrado. Execute sem --skip-download primeiro.")
                sys.exit(1)
            df_raw = pd.read_csv(raw_path)
            logger.info(f"Usando cache: {len(df_raw):,} jogos")

        # Adiciona resultados reais da Copa 2026 (se houver)
        updater = LiveUpdater()
        live_df = updater.to_dataframe()
        if not live_df.empty:
            logger.info(f"Incorporando {len(live_df)} resultados reais da Copa 2026")
            df_raw = pd.concat([df_raw, live_df], ignore_index=True)

        # ── 2. LIMPEZA ────────────────────────────────────────────────────
        logger.info("🧹 ETAPA 2/5 — Limpeza e feature engineering")
        df = clean_results(df_raw)

        # Filtro de ano para treino
        df_train = df[df["date"].dt.year >= args.min_year].copy()
        logger.info(f"Dados de treino: {len(df_train):,} jogos ({args.min_year}–hoje)")

        save_processed(df_train, "matches_clean")
        mlflow.log_param("n_matches_train", len(df_train))
        mlflow.log_param("min_year", args.min_year)

        # ── 3. ELO ────────────────────────────────────────────────────────
        elo = None
        if args.model in ("elo", "all"):
            logger.info("⚡ ETAPA 3/5 — Elo Rating System")
            elo = EloRatingSystem()
            elo.fit(df_train)
            elo.save(Path(__file__).parent / "outputs" / "models" / "elo_ratings.json")

            top10 = elo.top_teams(10)
            logger.info("Top 10 Elo após treino:")
            for rank, (team, rating) in enumerate(top10, 1):
                logger.info(f"  {rank:2}. {team:<25} {rating:.0f}")

            mlflow.log_metric("n_teams_elo", len(elo.ratings))
            mlflow.log_metric("top1_elo", top10[0][1] if top10 else 0)

        # ── 4. DIXON-COLES ────────────────────────────────────────────────
        dc = None
        if args.model in ("dc", "all"):
            logger.info("🧮 ETAPA 4/5 — Dixon-Coles (MLE)")
            dc = DixonColesModel()
            dc.fit(df_train, min_matches=5)

            mlflow.log_param("dc_home_adv", round(dc.home_adv, 3))
            mlflow.log_param("dc_rho",      round(dc.rho, 4))
            mlflow.log_metric("n_teams_dc", len(dc.teams))

        # ── 5. PREDIÇÕES COPA 2026 ────────────────────────────────────────
        logger.info("🔮 ETAPA 5/5 — Predições Copa 2026")

        if dc is None and elo is None:
            logger.error("Nenhum modelo disponível para predição")
            sys.exit(1)

        model_to_use = dc if dc is not None else None

        matches = [{"home": g["home"], "away": g["away"]} for g in COPA_2026_SCHEDULE]

        if model_to_use:
            predictions = model_to_use.predict_batch(matches, neutral=True)
        else:
            # Fallback: Elo puro
            rows = []
            for m in matches:
                pred = elo.predict(m["home"], m["away"], neutral=True)
                rows.append({
                    "home": m["home"], "away": m["away"],
                    "p_home_win": pred["p_home_win"],
                    "p_draw": pred["p_draw"],
                    "p_away_win": pred["p_away_win"],
                    "xg_home": pred["expected_goals_home"],
                    "xg_away": pred["expected_goals_away"],
                    "most_likely_score": "1–0",
                    "confidence": "N/A",
                })
            predictions = pd.DataFrame(rows)

        # ── RELATÓRIO ─────────────────────────────────────────────────────
        reporter = ReportGenerator(predictions)

        print(reporter.full_predictions_table())
        print("\n" + "─"*70)
        print(reporter.linkedin_post(focus_team="Brasil"))

        saved = reporter.save_all()
        for name, path in saved.items():
            mlflow.log_artifact(str(path))

        mlflow.log_metric("n_matches_predicted", len(predictions))

        logger.success(
            f"\n✅ Pipeline concluído! "
            f"{len(predictions)} predições geradas.\n"
            f"   Relatório: {saved.get('txt', '')}\n"
            f"   LinkedIn:  {saved.get('linkedin', '')}"
        )

    return predictions


if __name__ == "__main__":
    run()
