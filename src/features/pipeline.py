"""
CopIA — Pipeline de Feature Engineering
Orquestra todas as features e gera o dataset final para modelagem.
"""

from pathlib import Path

import pandas as pd
from loguru import logger

from src.features.form_features import build_form_matrix, compute_form_matchup_features
from src.features.h2h_features import compute_h2h_features
from src.features.ranking_features import (
    build_ranking_features,
    compute_ranking_matchup_features,
    get_all_teams_ranking,
)

ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"

# Confederações para encoding
CONFEDERATION_ENCODING = {
    "CONMEBOL": 0, "UEFA": 1, "CONCACAF": 2,
    "CAF": 3, "AFC": 4, "OFC": 5, "Desconhecido": 6,
}

# Dados estáticos dos times da Copa 2026
TEAMS_META = {
    "Brasil":       {"confederation": "CONMEBOL", "world_cup_titles": 5},
    "Argentina":    {"confederation": "CONMEBOL", "world_cup_titles": 3},
    "França":       {"confederation": "UEFA",     "world_cup_titles": 2},
    "Alemanha":     {"confederation": "UEFA",     "world_cup_titles": 4},
    "Espanha":      {"confederation": "UEFA",     "world_cup_titles": 1},
    "Inglaterra":   {"confederation": "UEFA",     "world_cup_titles": 1},
    "Portugal":     {"confederation": "UEFA",     "world_cup_titles": 0},
    "Itália":       {"confederation": "UEFA",     "world_cup_titles": 4},
    "Países Baixos":{"confederation": "UEFA",     "world_cup_titles": 0},
    "Croácia":      {"confederation": "UEFA",     "world_cup_titles": 0},
    "Uruguai":      {"confederation": "CONMEBOL", "world_cup_titles": 2},
    "Colômbia":     {"confederation": "CONMEBOL", "world_cup_titles": 0},
    "Marrocos":     {"confederation": "CAF",      "world_cup_titles": 0},
    "Senegal":      {"confederation": "CAF",      "world_cup_titles": 0},
    "Japão":        {"confederation": "AFC",      "world_cup_titles": 0},
    "Coreia do Sul":{"confederation": "AFC",      "world_cup_titles": 0},
    "Estados Unidos":{"confederation": "CONCACAF","world_cup_titles": 0},
    "México":       {"confederation": "CONCACAF", "world_cup_titles": 0},
}


def build_matchup_features(
    results: pd.DataFrame,
    team_a: str,
    team_b: str,
    reference_date: str = "2026-06-11",
    neutral_venue: bool = True,
) -> dict:
    """
    Constrói o vetor completo de features para um confronto entre dois times.
    Este é o input do modelo de predição.

    Args:
        results: DataFrame de resultados históricos
        team_a: Time "casa" (ou time A em campo neutro)
        team_b: Time "visitante" (ou time B)
        reference_date: Data do jogo
        neutral_venue: Se o jogo é em campo neutro

    Returns:
        Dict com todas as features do confronto
    """
    # 1. Features de ranking
    ranking_df = get_all_teams_ranking()
    ranking_features = compute_ranking_matchup_features(team_a, team_b, ranking_df)

    # 2. Features de forma
    form_a = build_form_matrix(results, [team_a], reference_date).to_dict("index").get(team_a, {})
    form_b = build_form_matrix(results, [team_b], reference_date).to_dict("index").get(team_b, {})
    form_features = compute_form_matchup_features(form_a, form_b)

    # 3. Features H2H
    h2h = compute_h2h_features(results, team_a, team_b, reference_date)
    h2h_features = {f"h2h_{k}" if not k.startswith("h2h_") else k: v
                    for k, v in h2h.items()
                    if not isinstance(v, str)}  # exclui strings

    # 4. Features contextuais
    meta_a = TEAMS_META.get(team_a, {"confederation": "Desconhecido", "world_cup_titles": 0})
    meta_b = TEAMS_META.get(team_b, {"confederation": "Desconhecido", "world_cup_titles": 0})

    context_features = {
        "neutral_venue": int(neutral_venue),
        "confederation_a": CONFEDERATION_ENCODING.get(meta_a["confederation"], 6),
        "confederation_b": CONFEDERATION_ENCODING.get(meta_b["confederation"], 6),
        "same_confederation": int(meta_a["confederation"] == meta_b["confederation"]),
        "titles_diff": meta_a["world_cup_titles"] - meta_b["world_cup_titles"],
        "titles_a": meta_a["world_cup_titles"],
        "titles_b": meta_b["world_cup_titles"],
    }

    # 5. Combina tudo
    all_features = {
        "team_a": team_a,
        "team_b": team_b,
        "reference_date": reference_date,
        **ranking_features,
        **form_features,
        **h2h_features,
        **context_features,
    }

    return all_features


def build_training_dataset(
    results: pd.DataFrame,
    min_year: int = 2000,
    reference_date: str = "2026-06-11",
) -> pd.DataFrame:
    """
    Constrói dataset de treino com features + target para todos os jogos históricos.

    Cada linha = um jogo histórico
    Target: 0=derrota_home, 1=empate, 2=vitória_home

    Args:
        results: DataFrame de resultados históricos
        min_year: Filtrar jogos a partir deste ano (mais recentes = mais relevantes)
        reference_date: Data de referência para calcular features

    Returns:
        DataFrame pronto para treino do modelo
    """
    logger.info("Construindo dataset de treino...")

    # Filtra jogos a partir do ano mínimo
    train_data = results[results["date"].dt.year >= min_year].copy()
    logger.info(f"Jogos a processar: {len(train_data):,}")

    rows = []
    errors = 0

    for idx, (_, game) in enumerate(train_data.iterrows()):
        if idx % 500 == 0:
            logger.debug(f"Processando jogo {idx}/{len(train_data)}...")

        try:
            home = game["home_team"]
            away = game["away_team"]
            game_date = str(game["date"].date())

            # Features calculadas com dados disponíveis ANTES do jogo
            features = build_matchup_features(
                results=results[results["date"] < game["date"]],
                team_a=home,
                team_b=away,
                reference_date=game_date,
                neutral_venue=game.get("neutral", False),
            )

            # Target
            if game["home_score"] > game["away_score"]:
                outcome = 2  # Vitória home
            elif game["home_score"] == game["away_score"]:
                outcome = 1  # Empate
            else:
                outcome = 0  # Derrota home

            features["outcome"] = outcome
            features["home_score"] = game["home_score"]
            features["away_score"] = game["away_score"]
            features["game_date"] = game_date

            rows.append(features)

        except Exception as e:
            errors += 1
            if errors <= 5:
                logger.warning(f"Erro no jogo {home} vs {away}: {e}")

    df = pd.DataFrame(rows)
    logger.success(
        f"Dataset criado: {len(df):,} jogos, {df.shape[1]} colunas, {errors} erros"
    )

    # Salva
    output_path = PROCESSED_DIR / "training_dataset.parquet"
    df.to_parquet(output_path, index=False)
    logger.success(f"Dataset salvo: {output_path}")

    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Retorna apenas as colunas numéricas usadas como features (exclui metadata)."""
    exclude = {
        "team_a", "team_b", "reference_date", "outcome",
        "home_score", "away_score", "game_date",
        "h2h_last_result",
    }
    return [c for c in df.columns if c not in exclude and df[c].dtype in ["float64", "int64", "int32"]]


if __name__ == "__main__":
    from src.data.ingestion import run_ingestion_pipeline

    data = run_ingestion_pipeline()
    results = data["international_results"]

    # Exemplo: features para Brasil vs. Argentina
    features = build_matchup_features(results, "Brasil", "Argentina")
    print("\nFeatures Brasil vs. Argentina:")
    for k, v in features.items():
        if isinstance(v, (int, float)):
            print(f"  {k}: {v:.4f}")
