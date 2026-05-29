# ⚽ CopIA — Score Predictor

> Subprojeto do [CopIA](../README.md) focado em **predição de placares** da Copa do Mundo 2026.
> Modelos treinados em 40.000+ jogos históricos reais.

---

## O que este módulo faz

| Saída | Exemplo |
|---|---|
| Placar mais provável | Brasil 1–0 Nigéria |
| Probabilidades W/D/L | Vitória 52% · Empate 27% · Derrota 21% |
| xG esperado por time | Brasil 1.48 · Nigéria 0.82 |
| Top 8 cenários de placar | com probabilidade individual |
| Nível de confiança | "Favorito" / "Imprevisível" |
| Relatório LinkedIn | post pronto após cada rodada |

---

## Modelos Disponíveis

### 1. Elo Rating System (`src/models/elo.py`)
- K-factor dinâmico por competição (Copa=60, Amistoso=20)
- Ajuste de placar (*goal difference multiplier*)
- Probabilidade de empate calibrada empiricamente
- 100% interpretável

### 2. Dixon-Coles (`src/models/dixon_coles.py`)
- Parâmetros α (ataque) e β (defesa) **aprendidos de dados reais**
- Correção τ para placares baixos (0-0, 1-0, 0-1, 1-1)
- Fator γ de home advantage ajustado
- Matriz completa de placares (probabilidade de cada 0-0 até 8-8)

### 3. Ensemble ML (`src/models/ensemble.py`) *(em desenvolvimento)*
- XGBoost + LightGBM com 20 features reais
- Meta-learner calibrado (Isotonic Regression)
- Feature importance via SHAP

---

## Quick Start

```bash
# Da raiz do copa-ia/
cd score-predictor

# Pipeline completo (download + treino + predições)
uv run python run_pipeline.py

# Apenas com cache local (sem re-download)
uv run python run_pipeline.py --skip-download

# Rodar testes
uv run pytest tests/ -v
```

---

## Adicionar Resultados Reais (Durante a Copa)

```python
from src.data.live_updater import LiveUpdater

updater = LiveUpdater()

# Após cada jogo:
updater.add_result("Brasil", "Nigéria", 2, 0, "2026-06-17")
updater.add_result("Japão",  "Costa do Marfim", 1, 2, "2026-06-13")

# Ver o que já foi registrado:
updater.show_results()

# Re-rodar o pipeline com os novos dados:
# uv run python run_pipeline.py --skip-download
```

---

## Estrutura

```
score-predictor/
├── run_pipeline.py          # Script principal
├── src/
│   ├── data/
│   │   ├── downloader.py    # Download de 40k+ jogos históricos
│   │   ├── preprocessor.py  # Limpeza + pesos por competição/tempo
│   │   └── live_updater.py  # Resultados reais da Copa 2026
│   ├── models/
│   │   ├── elo.py           # Elo dinâmico com K-factor por competição
│   │   ├── dixon_coles.py   # Poisson bivariado ajustado por MLE
│   │   ├── ensemble.py      # XGBoost + LightGBM (em dev)
│   │   └── evaluator.py     # RPS · Brier · Acurácia · MAE
│   └── reporting/
│       └── generator.py     # Relatórios e posts LinkedIn
├── data/
│   ├── raw/                 # CSVs baixados (gitignored)
│   ├── processed/           # Parquets limpos
│   └── live/                # Resultados reais Copa 2026
├── outputs/
│   ├── models/              # Modelos treinados
│   ├── predictions/         # Predições geradas
│   └── reports/             # Relatórios e posts
├── notebooks/               # Notebooks didáticos (em desenvolvimento)
└── tests/                   # 20 testes automatizados
```

---

## Métricas de Avaliação

| Métrica | O que mede | Referência |
|---|---|---|
| **RPS** | Quão bem calibradas são as probabilidades | Padrão sports analytics |
| **Brier Score** | Erro quadrático nas probabilidades | Menor = melhor |
| **Acurácia W/D/L** | % de resultados corretos | Baseline: ~50% |
| **Placar exato** | % de placares corretos | Baseline: ~5–8% |
| **MAE gols** | Erro médio no número de gols | Típico: 0.8–1.2 |

---

## Dados

- **Fonte principal:** [martj42/international_results](https://github.com/martj42/international_results)
  (~47.000 jogos desde 1872, atualizado regularmente)
- **Treino:** jogos a partir de 2000 (padrão)
- **Validação:** Copa do Mundo 2022
- **Live:** Copa 2026 via `LiveUpdater`
