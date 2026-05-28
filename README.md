# ⚽ CopIA — Inteligência Artificial aplicada à Copa do Mundo 2026

> **ML · Simulação Monte Carlo · Reinforcement Learning · GenAI**

[![CI](https://github.com/ahirtonlopes/copa-ia/actions/workflows/ci.yml/badge.svg)](https://github.com/ahirtonlopes/copa-ia/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ahirtonlopes/copa-ia/blob/main/notebooks/fase1/01_exploratory_data_analysis.ipynb)

**CopIA** é um projeto de portfólio técnico e material didático que aplica IA moderna ao contexto da Copa do Mundo 2026. O projeto cobre o espectro completo — de ML clássico a agentes de Reinforcement Learning — usando dados reais de futebol.

---

## 🚀 Quick Start

```bash
# 1. Clone o repositório
git clone https://github.com/ahirtonlopes/copa-ia.git
cd copa-ia

# 2. Instale as dependências (requer UV)
make install

# 3. Configure as variáveis de ambiente
cp .env.example .env
# Edite .env com suas chaves de API (opcional para Fase 1)

# 4. Baixe os dados
make download-data

# 5. Rode o dashboard
make run-app
```

Ou direto no Colab: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ahirtonlopes/copa-ia/blob/main/notebooks/fase1/01_exploratory_data_analysis.ipynb)

---

## 📦 Estrutura do Projeto

```
copa-ia/
├── notebooks/
│   ├── fase1/          # ML, feature engineering, simulação
│   └── fase2/          # Reinforcement Learning tático
├── src/
│   ├── data/           # Ingestão e preprocessamento
│   ├── features/       # Feature engineering (ranking, forma, H2H)
│   ├── models/         # Baseline (Poisson) + Ensemble ML
│   ├── simulation/     # Motor Monte Carlo
│   ├── rl/             # Ambiente Gymnasium + agentes RL
│   └── agent/          # Agente conversacional (LangGraph)
├── app/                # Dashboard Streamlit
└── tests/              # Testes automatizados
```

---

## 🎯 Fases do Projeto

### Fase 1 — Previsão e Simulação (✅ Disponível)

| Componente | Descrição |
|---|---|
| **Feature Engineering** | 47+ features: ranking FIFA, forma recente, H2H, valor de mercado |
| **Modelo Baseline** | Poisson Bivariado com correção Dixon-Coles |
| **Ensemble ML** | XGBoost + LightGBM + Random Forest com stacking |
| **Monte Carlo** | 100.000 simulações do torneio completo |
| **Explainability** | SHAP values + narrativa em linguagem natural |
| **Dashboard** | Streamlit interativo com bracket e probabilidades |

### Fase 2 — Agente Tático com RL (🔄 Em desenvolvimento)

| Componente | Descrição |
|---|---|
| **Gymnasium Env** | Ambiente customizado para decisões de substituição |
| **Q-Learning** | Implementação tabular didática |
| **DQN** | Deep Q-Network com PyTorch |
| **PPO** | Proximal Policy Optimization via Stable-Baselines3 |
| **Visualização** | Decision heatmap, episode replay, comparação com humanos |

---

## 📊 Resultados Preliminares

> **Copa do Mundo 2026 — Simulação com 100.000 cenários**

| # | Seleção | P(Campeão) | P(Final) | P(Semi) |
|---|---|---|---|---|
| 1 | Argentina | ~18% | ~34% | ~52% |
| 2 | França | ~16% | ~31% | ~49% |
| 3 | Brasil | ~13% | ~26% | ~44% |
| 4 | Espanha | ~11% | ~22% | ~40% |
| 5 | Inglaterra | ~9% | ~19% | ~36% |

*Probabilidades variam com a simulação. Execute localmente para resultados atualizados.*

---

## 📚 Notebooks Didáticos

| Notebook | Conceito | Nível |
|---|---|---|
| `01_exploratory_data_analysis` | EDA, visualização, estatísticas | Iniciante |
| `02_feature_engineering` | Feature extraction, janelas deslizantes | Intermediário |
| `03_baseline_models` | Regressão Logística, Poisson | Iniciante |
| `04_ensemble_ml` | XGBoost, LightGBM, stacking | Intermediário |
| `05_calibration_explainability` | Platt Scaling, SHAP | Avançado |
| `06_poisson_model` | Dixon-Coles, distribuição de gols | Intermediário |
| `07_monte_carlo_simulation` | Monte Carlo, simulação do torneio | Intermediário |
| `08_responsible_ai_bias` | Viés, fairness, limitações | Todos |
| `10_rl_introduction_mdp` | MDP, estados, ações, recompensas | Intermediário |
| `11_tactical_environment` | Gymnasium customizado | Avançado |
| `12_q_learning_agent` | Q-Learning tabular | Intermediário |
| `13_dqn_agent_pytorch` | Deep Q-Network | Avançado |
| `14_ppo_stable_baselines` | PPO, Stable-Baselines3 | Avançado |

---

## 🛠️ Stack Tecnológica

```
ML/DL:     scikit-learn · XGBoost · LightGBM · PyTorch · SHAP
RL:        Gymnasium · Stable-Baselines3
GenAI:     Anthropic Claude · LangGraph · ChromaDB
Dados:     StatsBombPy · SoccerData · pandas · numpy
Dashboard: Streamlit · Plotly
MLOps:     MLflow · pytest · ruff
```

---

## 🤝 Como Contribuir

1. Fork o repositório
2. Crie uma branch: `git checkout -b feature/minha-feature`
3. Commit: `git commit -m 'feat: adiciona minha feature'`
4. Push: `git push origin feature/minha-feature`
5. Abra um Pull Request

Veja [CONTRIBUTING.md](CONTRIBUTING.md) para detalhes.

---

## 📄 Licença

MIT License — veja [LICENSE](LICENSE) para detalhes.

---

## 👤 Autor

**Ahirton Lopes**
- PhD/MSc em IA — Universidade Presbiteriana Mackenzie
- Professor de IA/ML — FIAP
- Google Developer Expert (GDE)
- Microsoft MVP

[![LinkedIn](https://img.shields.io/badge/LinkedIn-ahirtonlopes-blue)](https://linkedin.com/in/ahirtonlopes)
[![GitHub](https://img.shields.io/badge/GitHub-ahirtonlopes-black)](https://github.com/ahirtonlopes)

---

> *"A Copa do Mundo é o melhor dataset do mundo para aprender IA — porque todo mundo quer saber o resultado."*
