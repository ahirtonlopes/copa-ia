<div align="center">

# ⚽ CopIA

### Inteligência Artificial aplicada à Copa do Mundo 2026

*De regressão logística a agentes de Reinforcement Learning — usando o torneio mais assistido do planeta como laboratório de IA*

[![CI](https://github.com/ahirtonlopes/copa-ia/actions/workflows/ci.yml/badge.svg)](https://github.com/ahirtonlopes/copa-ia/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ahirtonlopes/copa-ia/blob/main/notebooks/fase1/01_exploratory_data_analysis.ipynb)
[![Stars](https://img.shields.io/github/stars/ahirtonlopes/copa-ia?style=social)](https://github.com/ahirtonlopes/copa-ia/stargazers)

**[Notebooks](#-notebooks-didáticos) · [Quick Start](#-quick-start) · [Resultados](#-100000-simulações-da-copa-2026) · [Arquitetura](#️-arquitetura) · [Roadmap](#️-roadmap)**

</div>

---

## O que é o CopIA?

**CopIA** é um projeto de portfólio técnico e material didático em português que aplica o espectro completo de IA moderna a um problema que todo mundo entende: *quem vai ganhar a Copa do Mundo 2026?*

O projeto cresce em duas fases:

| Fase | Status | O que cobre |
|---|---|---|
| **Fase 1 — Predict** | ✅ Disponível | Feature engineering · Modelos ML · Poisson · Monte Carlo · SHAP · Dashboard |
| **Fase 2 — Tactics** | 🔄 Em build | Gymnasium env · Q-Learning · DQN · PPO · Visualização de comportamento |

> **Por que futebol?**
> Porque quando você explica Monte Carlo com "simulamos o torneio 100.000 vezes", qualquer pessoa entende o que está acontecendo — e quer saber o resultado.

---

## 🎲 100.000 Simulações da Copa 2026

> Resultados gerados pelo modelo Poisson + 100k iterações Monte Carlo · Atualizado em 28/05/2026

```
  #   Seleção              Campeão    Final    Semi    Quartas
 ─────────────────────────────────────────────────────────────
  1   🇦🇷 Argentina          9.2%    14.4%   43.5%    65.2%
  2   🇫🇷 França             8.0%    12.7%   40.3%    62.0%
  3   🇧🇷 Brasil             6.6%    11.3%   37.2%    59.4%
  4   🇪🇸 Espanha            6.0%    10.5%   34.7%    56.4%
  5   🏴󠁧󠁢󠁥󠁮󠁧󠁿 Inglaterra        5.3%     9.7%   33.2%    55.8%
  6   🇩🇪 Alemanha           5.1%     9.0%   32.2%    55.1%
  7   🇵🇹 Portugal           5.1%     9.0%   31.5%    53.1%
  8   🇳🇱 Países Baixos      4.5%     8.3%   29.5%    51.9%
  9   🇧🇪 Bélgica            3.7%     7.0%   25.6%    46.4%
 10   🇨🇴 Colômbia           3.2%     6.4%   24.8%    46.3%
 11   🇲🇦 Marrocos           2.9%     5.9%   24.0%    46.0%
 12   🇩🇰 Dinamarca          2.8%     5.8%   22.8%    43.5%
```

<details>
<summary>Ver tabela completa (todos os times)</summary>

```
  #   Seleção              Campeão    Final    Semi    Quartas  Oitavas
 ──────────────────────────────────────────────────────────────────────
  1   Argentina             9.2%    14.4%   43.5%    65.2%    96.2%
  2   França                8.0%    12.7%   40.3%    62.0%    93.5%
  3   Brasil                6.6%    11.3%   37.2%    59.4%    92.9%
  4   Espanha               6.0%    10.5%   34.7%    56.4%    90.3%
  5   Inglaterra            5.3%     9.7%   33.2%    55.8%    91.2%
  6   Alemanha              5.1%     9.0%   32.2%    55.1%    92.1%
  7   Portugal              5.1%     9.0%   31.5%    53.1%    87.2%
  8   Países Baixos         4.5%     8.3%   29.5%    51.9%    88.6%
  9   Bélgica               3.7%     7.0%   25.6%    46.4%    81.8%
 10   Colômbia              3.2%     6.4%   24.8%    46.3%    84.1%
 11   Marrocos              2.9%     5.9%   24.0%    46.0%    86.6%
 12   Dinamarca             2.8%     5.8%   22.8%    43.5%    80.5%
 13   Croácia               2.6%     5.3%   21.0%    40.0%    74.2%
 14   Japão                 2.4%     5.1%   20.3%    40.1%    76.7%
 15   Estados Unidos        2.2%     4.9%   20.6%    42.4%    84.3%
 16   Uruguai               2.3%     4.6%   18.1%    35.0%    64.8%
 17   Coreia do Sul         2.0%     4.5%   19.1%    39.5%    79.3%
 18   Sérvia                2.0%     4.2%   17.4%    35.1%    67.9%
 19   Senegal               1.9%     4.3%   18.2%    37.4%    74.6%
 20   México                1.9%     4.3%   18.6%    39.5%    80.9%
```
</details>

**Reproduza em segundos:**
```bash
uv run python -c "
from src.models.baseline import PoissonModel
from src.simulation.monte_carlo import TournamentSimulator
sim = TournamentSimulator(model=PoissonModel())
sim.run(n_simulations=100_000).print_summary()
"
```

---

## 🚀 Quick Start

### Pré-requisitos
- Python 3.11+
- [UV](https://docs.astral.sh/uv/) — `curl -LsSf https://astral.sh/uv/install.sh | sh`

```bash
# 1. Clone
git clone https://github.com/ahirtonlopes/copa-ia.git
cd copa-ia

# 2. Instale tudo (cria .venv automaticamente)
make install

# 3. Configure suas APIs (opcional — Fase 1 roda sem API key)
cp .env.example .env

# 4. Rode a simulação
make simulate

# 5. Suba o dashboard
make run-app
```

**Ou direto no navegador, sem instalar nada:**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ahirtonlopes/copa-ia/blob/main/notebooks/fase1/01_exploratory_data_analysis.ipynb)

---

## 📚 Notebooks Didáticos

Cada notebook é autossuficiente — roda no Colab sem configuração local.

### Fase 1 — Predição e Simulação

| # | Notebook | Conceito ensinado | Nível | Colab |
|---|---|---|---|---|
| 01 | `01_exploratory_data_analysis` | EDA, visualização, estatísticas descritivas | 🟢 Iniciante | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ahirtonlopes/copa-ia/blob/main/notebooks/fase1/01_exploratory_data_analysis.ipynb) |
| 02 | `02_feature_engineering` | Feature extraction, janelas deslizantes, decay exponencial | 🟡 Intermediário | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ahirtonlopes/copa-ia/blob/main/notebooks/fase1/02_feature_engineering.ipynb) |
| 03 | `03_baseline_models` | Regressão Logística, avaliação de classificação | 🟢 Iniciante | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ahirtonlopes/copa-ia/blob/main/notebooks/fase1/03_baseline_models.ipynb) |
| 04 | `04_ensemble_ml` | XGBoost, LightGBM, stacking | 🟡 Intermediário | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ahirtonlopes/copa-ia/blob/main/notebooks/fase1/04_ensemble_ml.ipynb) |
| 05 | `05_calibration_explainability` | Platt Scaling, SHAP waterfall | 🔴 Avançado | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ahirtonlopes/copa-ia/blob/main/notebooks/fase1/05_calibration_explainability.ipynb) |
| 06 | `06_poisson_model` | Distribuição de Poisson, Dixon-Coles | 🟡 Intermediário | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ahirtonlopes/copa-ia/blob/main/notebooks/fase1/06_poisson_model.ipynb) |
| 07 | `07_monte_carlo_simulation` | Monte Carlo, simulação do torneio completo | 🟡 Intermediário | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ahirtonlopes/copa-ia/blob/main/notebooks/fase1/07_monte_carlo_simulation.ipynb) |
| 08 | `08_responsible_ai_bias` | Viés, fairness, limitações de modelos esportivos | 🟢 Todos | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ahirtonlopes/copa-ia/blob/main/notebooks/fase1/08_responsible_ai_bias.ipynb) |

### Fase 2 — Agente Tático com RL *(em desenvolvimento)*

| # | Notebook | Conceito ensinado | Nível |
|---|---|---|---|
| 10 | `10_rl_introduction_mdp` | MDP, estados, ações, recompensas com futebol | 🟡 Intermediário |
| 11 | `11_tactical_environment` | Gymnasium custom env para substituições | 🔴 Avançado |
| 12 | `12_q_learning_agent` | Q-Learning tabular — do zero ao agente | 🟡 Intermediário |
| 13 | `13_dqn_agent_pytorch` | Deep Q-Network com PyTorch | 🔴 Avançado |
| 14 | `14_ppo_stable_baselines` | PPO com Stable-Baselines3 | 🔴 Avançado |
| 15 | `15_agent_visualization` | Decision heatmap, episode replay | 🔴 Avançado |
| 16 | `16_human_vs_agent` | Agente vs. decisão de treinadores reais | 🔴 Avançado |

---

## 🏗️ Arquitetura

```
                        ┌─────────────────────────────┐
                        │       Fontes de Dados        │
                        │  Kaggle · StatsBomb · FBref  │
                        │  FIFA Ranking · Transfermkt  │
                        └──────────────┬──────────────┘
                                       │
                        ┌──────────────▼──────────────┐
                        │     Feature Engineering      │
                        │  ranking · forma · H2H       │
                        │  elenco · contexto · xG      │
                        └──────────────┬──────────────┘
                                       │
               ┌───────────────────────┼───────────────────────┐
               │                       │                       │
  ┌────────────▼──────────┐ ┌──────────▼──────────┐ ┌────────▼────────────┐
  │   Modelos Preditivos   │ │  Motor de Simulação  │ │   Agente RL (F2)   │
  │  Poisson · XGBoost    │ │  Monte Carlo 100k    │ │  Gymnasium env      │
  │  LightGBM · Ensemble  │ │  Bracket · Cenários  │ │  Q-Learning · PPO  │
  └────────────┬──────────┘ └──────────┬──────────┘ └────────┬────────────┘
               │                       │                       │
               └───────────────────────┼───────────────────────┘
                                       │
                        ┌──────────────▼──────────────┐
                        │    Explainability + GenAI    │
                        │   SHAP · LangGraph · RAG     │
                        │  Claude · Gemini · GPT-4o   │
                        └──────────────┬──────────────┘
                                       │
                        ┌──────────────▼──────────────┐
                        │     Dashboard Streamlit      │
                        │  Predições · Bracket · Chat  │
                        └─────────────────────────────┘
```

---

## 🛠️ Stack Tecnológica

```
Área              Ferramentas
────────────────────────────────────────────────────────────
ML / Stats        scikit-learn · XGBoost · LightGBM · scipy
Deep Learning     PyTorch
Explicabilidade   SHAP · LIME
RL                Gymnasium · Stable-Baselines3
GenAI / Agents    Anthropic Claude · LangGraph · ChromaDB
                  OpenAI · Gemini (suporte configurável)
Dados             pandas · numpy · StatsBombPy · SoccerData
Visualização      Streamlit · Plotly · matplotlib · seaborn
MLOps             MLflow · pytest · ruff · GitHub Actions
Empacotamento     UV · hatchling
```

---

## 📁 Estrutura do Repositório

```
copa-ia/
├── notebooks/
│   ├── fase1/          # 8 notebooks: EDA → SHAP → Monte Carlo
│   └── fase2/          # 7 notebooks: MDP → Q-Learning → PPO
├── src/
│   ├── data/           # Ingestão e preprocessamento
│   ├── features/       # Feature engineering (ranking, forma, H2H)
│   ├── models/         # Baseline (Poisson) + Ensemble ML
│   ├── simulation/     # Motor Monte Carlo (100k sims em ~17s)
│   ├── rl/             # Ambiente Gymnasium + agentes RL
│   └── agent/          # Agente conversacional (LangGraph)
├── app/                # Dashboard Streamlit multi-página
├── tests/              # 55 testes automatizados
├── data/sample/        # Dados de amostra para Colab
└── docs/workshops/     # Guias de workshop e palestra
```

---

## 🗺️ Roadmap

```
✅ Fase 1a  — Motor Poisson + Monte Carlo (100k sims em ~17s)
✅ Fase 1b  — Estrutura de projeto, 55 testes, CI/CD
🔄 Fase 1c  — Ensemble ML (XGBoost + LightGBM) + SHAP
🔄 Fase 1d  — Dashboard Streamlit público
🔄 Fase 1e  — Notebooks didáticos (01–08) com Colab badges
⏳ Fase 2a  — Gymnasium env (TacticalCupEnv)
⏳ Fase 2b  — Q-Learning tabular + visualização
⏳ Fase 2c  — PPO via Stable-Baselines3
⏳ Fase 2d  — Agente vs. decisão humana real (StatsBomb)
⏳ Futuro   — Multi-agent RL · Computer Vision · Live data
```

---

## 🧪 Testes

```bash
make test          # Roda todos os 55 testes
make test-cov      # Com relatório de cobertura
make lint          # ruff check + ruff format
```

```
tests/test_models.py       ·····························  29 passed
tests/test_rl_env.py       ··············                 14 passed
tests/test_simulation.py   ············                   12 passed
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
55 passed in 2.06s
```

---

## 🤝 Contribuindo

Contribuições são bem-vindas! Veja [CONTRIBUTING.md](CONTRIBUTING.md) para o fluxo completo.

Formas rápidas de contribuir:
- ⭐ Dê uma estrela se o projeto te ajudou
- 🐛 Reporte bugs via [Issues](https://github.com/ahirtonlopes/copa-ia/issues)
- 💡 Sugira melhorias ou novas features
- 📓 Melhore os notebooks didáticos
- 🌍 Ajude a traduzir para o inglês

---

## 📄 Licença

MIT License — veja [LICENSE](LICENSE) para detalhes.

---

## 👤 Autor

<div align="center">

**Ahirton Lopes**

PhD/MSc em IA · Universidade Presbiteriana Mackenzie
Professor de IA/ML · FIAP
Google Developer Expert (GDE) · Microsoft MVP

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Ahirton_Lopes-0077B5?logo=linkedin)](https://linkedin.com/in/ahirtonlopes)
[![GitHub](https://img.shields.io/badge/GitHub-ahirtonlopes-181717?logo=github)](https://github.com/ahirtonlopes)

</div>

---

<div align="center">

*"A Copa do Mundo é o melhor dataset do mundo para aprender IA — porque todo mundo quer saber o resultado."*

**⭐ Se este projeto te ajudou a aprender, considera deixar uma estrela!**

</div>
