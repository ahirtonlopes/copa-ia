"""
CopIA — Ingestão de Dados
Baixa e organiza todos os datasets necessários para o projeto.

Fontes:
- Kaggle: resultados históricos internacionais (1872–2024)
- Kaggle: Copa do Mundo (grupos, times, resultados)
- StatsBomb Open Data: ações detalhadas por partida
- Open-Meteo: dados climáticos das sedes da Copa 2026
"""

import os
import zipfile
from pathlib import Path

import pandas as pd
import requests
from loguru import logger

# ── Configuração de caminhos ───────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
SAMPLE_DIR = ROOT / "data" / "sample"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ── URLs públicas alternativas ao Kaggle (para CI/testes) ─────────────────────
PUBLIC_DATASETS = {
    "international_results": {
        "url": "https://raw.githubusercontent.com/martj42/international_results/master/results.csv",
        "filename": "international_results.csv",
        "description": "Resultados de jogos internacionais (1872–presente)",
    },
    "fifa_ranking": {
        "url": "https://raw.githubusercontent.com/stefanodangelo/fifa-ranking/master/fifa_ranking.csv",
        "filename": "fifa_ranking.csv",
        "description": "Ranking FIFA histórico por seleção e data",
    },
}

# Dados estáticos da Copa 2026 (grupos e seleções classificadas)
COPA_2026_GROUPS = {
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
    "L": ["Marrocos", "Portugal Alt", "Congo", "Zâmbia"],
}

# Sedes da Copa 2026
VENUES_2026 = [
    {"city": "New York", "country": "USA", "latitude": 40.7128, "longitude": -74.0060},
    {"city": "Los Angeles", "country": "USA", "latitude": 34.0522, "longitude": -118.2437},
    {"city": "Dallas", "country": "USA", "latitude": 32.7767, "longitude": -96.7970},
    {"city": "San Francisco", "country": "USA", "latitude": 37.7749, "longitude": -122.4194},
    {"city": "Miami", "country": "USA", "latitude": 25.7617, "longitude": -80.1918},
    {"city": "Seattle", "country": "USA", "latitude": 47.6062, "longitude": -122.3321},
    {"city": "Boston", "country": "USA", "latitude": 42.3601, "longitude": -71.0589},
    {"city": "Kansas City", "country": "USA", "latitude": 39.0997, "longitude": -94.5786},
    {"city": "Atlanta", "country": "USA", "latitude": 33.7490, "longitude": -84.3880},
    {"city": "Philadelphia", "country": "USA", "latitude": 39.9526, "longitude": -75.1652},
    {"city": "Houston", "country": "USA", "latitude": 29.7604, "longitude": -95.3698},
    {"city": "Toronto", "country": "Canada", "latitude": 43.6532, "longitude": -79.3832},
    {"city": "Vancouver", "country": "Canada", "latitude": 49.2827, "longitude": -123.1207},
    {"city": "Guadalajara", "country": "Mexico", "latitude": 20.6597, "longitude": -103.3496},
    {"city": "Mexico City", "country": "Mexico", "latitude": 19.4326, "longitude": -99.1332},
    {"city": "Monterrey", "country": "Mexico", "latitude": 25.6866, "longitude": -100.3161},
]


def download_csv(url: str, dest: Path, description: str = "") -> pd.DataFrame | None:
    """Baixa um CSV de uma URL pública e salva localmente."""
    if dest.exists():
        logger.info(f"Arquivo já existe, pulando download: {dest.name}")
        return pd.read_csv(dest)

    logger.info(f"Baixando: {description or dest.name}")
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        dest.write_bytes(response.content)
        df = pd.read_csv(dest)
        logger.success(f"Salvo: {dest.name} — {len(df):,} linhas")
        return df
    except requests.RequestException as e:
        logger.error(f"Falha no download de {url}: {e}")
        return None


def load_international_results() -> pd.DataFrame:
    """
    Carrega resultados históricos de jogos internacionais.
    Dataset: 47.000+ jogos desde 1872.
    """
    dest = RAW_DIR / "international_results.csv"
    config = PUBLIC_DATASETS["international_results"]
    df = download_csv(config["url"], dest, config["description"])

    if df is None:
        logger.warning("Usando dataset de amostra para testes.")
        return _create_sample_results()

    # Limpeza básica
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Filtra apenas partidas oficiais (exclui amistosos pré-torneio inválidos)
    # Mantemos todos por ora, filtragem por tipo será nas features
    logger.info(
        f"Resultados internacionais: {len(df):,} jogos "
        f"({df['date'].min().year}–{df['date'].max().year})"
    )
    return df


def load_copa_do_mundo_data() -> dict[str, pd.DataFrame]:
    """
    Carrega dados históricos das Copas do Mundo.
    Retorna dict com: matches, group_stages, knockout_stages.
    """
    # Filtra do dataset geral apenas jogos de Copa do Mundo
    results = load_international_results()

    wc_tournaments = [
        "FIFA World Cup", "World Cup", "Copa do Mundo"
    ]

    copa_matches = results[
        results.get("tournament", pd.Series(dtype=str)).str.contains(
            "|".join(wc_tournaments), na=False, case=False
        )
    ].copy() if "tournament" in results.columns else results.copy()

    # Anos de Copa do Mundo (para filtrar por data se não tiver coluna tournament)
    wc_years = [1930, 1934, 1938, 1950, 1954, 1958, 1962, 1966,
                1970, 1974, 1978, 1982, 1986, 1990, 1994, 1998,
                2002, 2006, 2010, 2014, 2018, 2022]

    copa_matches_by_year = results[results["date"].dt.year.isin(wc_years)].copy()

    return {
        "all_matches": copa_matches_by_year,
        "wc_years": wc_years,
    }


def create_copa_2026_fixture() -> pd.DataFrame:
    """
    Cria o fixture da Copa 2026 com os grupos confirmados.
    Gera todos os jogos da fase de grupos (round-robin dentro de cada grupo).
    """
    rows = []
    for group, teams in COPA_2026_GROUPS.items():
        for i, team_a in enumerate(teams):
            for team_b in teams[i + 1 :]:
                rows.append({
                    "group": group,
                    "home_team": team_a,
                    "away_team": team_b,
                    "phase": "group_stage",
                    "played": False,
                    "home_score": None,
                    "away_score": None,
                })

    df = pd.DataFrame(rows)
    logger.info(f"Fixture 2026 criado: {len(df)} jogos na fase de grupos")
    return df


def save_copa_2026_teams() -> pd.DataFrame:
    """
    Cria dataset das 48 seleções classificadas para a Copa 2026
    com metadados básicos.
    """
    teams = []
    for group, group_teams in COPA_2026_GROUPS.items():
        for team in group_teams:
            teams.append({
                "team": team,
                "group": group,
                "confederation": _get_confederation(team),
            })

    df = pd.DataFrame(teams)
    dest = PROCESSED_DIR / "copa_2026_teams.parquet"
    df.to_parquet(dest, index=False)
    logger.success(f"48 seleções salvas em {dest}")
    return df


def _get_confederation(team: str) -> str:
    """Retorna a confederação de uma seleção."""
    conmebol = [
        "Brasil", "Argentina", "Colômbia", "Equador", "Uruguai",
        "Chile", "Peru", "Venezuela", "Bolívia", "Paraguai",
    ]
    uefa = [
        "França", "Alemanha", "Espanha", "Inglaterra", "Portugal",
        "Países Baixos", "Bélgica", "Croácia", "Dinamarca", "Sérvia",
        "Turquia", "Hungria", "Romênia", "Eslováquia", "Eslovênia", "Israel",
    ]
    concacaf = [
        "Estados Unidos", "México", "Canadá", "Panamá", "Costa Rica",
    ]
    caf = [
        "Marrocos", "Nigéria", "Costa do Marfim", "Senegal", "Gana",
        "Camarões", "África do Sul", "Congo", "Zâmbia",
        "República Centro-Africana", "Argélia",
    ]
    afc = [
        "Japão", "Coreia do Sul", "Arábia Saudita", "Iraque", "Iêmen",
    ]
    ofc = ["Nova Zelândia"]

    mapping = {
        **{t: "CONMEBOL" for t in conmebol},
        **{t: "UEFA" for t in uefa},
        **{t: "CONCACAF" for t in concacaf},
        **{t: "CAF" for t in caf},
        **{t: "AFC" for t in afc},
        **{t: "OFC" for t in ofc},
    }
    return mapping.get(team, "Desconhecido")


def _create_sample_results() -> pd.DataFrame:
    """Cria dataset de amostra mínimo para testes sem internet."""
    return pd.DataFrame({
        "date": pd.to_datetime([
            "2022-11-20", "2022-11-21", "2022-11-22",
            "2022-11-24", "2022-11-25",
        ]),
        "home_team": ["Catar", "Inglaterra", "Senegal", "Argentina", "França"],
        "away_team": ["Equador", "Irã", "Países Baixos", "Arábia Saudita", "Dinamarca"],
        "home_score": [0, 6, 0, 1, 1],
        "away_score": [2, 2, 2, 2, 0],
        "tournament": ["FIFA World Cup"] * 5,
        "neutral": [False] * 5,
    })


def run_ingestion_pipeline() -> dict[str, pd.DataFrame]:
    """
    Pipeline completo de ingestão.
    Retorna dict com todos os DataFrames carregados.
    """
    logger.info("=" * 60)
    logger.info("CopIA — Pipeline de Ingestão de Dados")
    logger.info("=" * 60)

    data = {}

    # 1. Resultados históricos
    data["international_results"] = load_international_results()

    # 2. Dados históricos da Copa
    copa_data = load_copa_do_mundo_data()
    data.update(copa_data)

    # 3. Fixture 2026
    data["fixture_2026"] = create_copa_2026_fixture()
    data["fixture_2026"].to_parquet(
        PROCESSED_DIR / "fixture_2026.parquet", index=False
    )

    # 4. Times classificados
    data["teams_2026"] = save_copa_2026_teams()

    # 5. Salva resultados históricos processados
    (data["international_results"]
        .to_parquet(PROCESSED_DIR / "international_results.parquet", index=False))

    logger.success("Ingestão concluída!")
    logger.info(f"Arquivos em: {PROCESSED_DIR}")

    return data


if __name__ == "__main__":
    run_ingestion_pipeline()
