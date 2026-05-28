"""
CopIA — Dashboard Principal
Streamlit app com navegação multi-página para o projeto CopIA.
"""

import streamlit as st

# ── Configuração da página ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="CopIA — Copa do Mundo 2026 com IA",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/football2.png", width=80)
    st.title("CopIA ⚽")
    st.caption("Copa do Mundo 2026 × Inteligência Artificial")

    st.markdown("---")
    st.markdown("**Navegação**")
    st.markdown("""
    - 🏆 Probabilidades de Campeão
    - 📊 Bracket Interativo
    - 🔬 Explainability (SHAP)
    - 🎲 Simulação Monte Carlo
    - 🤖 Agente Tático (RL)
    - 💬 Chat com CopIA
    """)

    st.markdown("---")
    st.markdown("**Sobre o Projeto**")
    st.markdown("""
    CopIA combina **ML, Simulação Monte Carlo e Reinforcement Learning**
    para analisar e prever a Copa do Mundo 2026.

    Desenvolvido por [Ahirton Lopes](https://github.com/ahirtonlopes)
    como projeto de portfólio técnico e material didático.
    """)

    st.markdown("---")
    with st.expander("⚙️ Configurações da Simulação"):
        n_sims = st.slider("Nº de simulações", 1_000, 100_000, 10_000, step=1_000)
        st.session_state["n_simulations"] = n_sims

        seed = st.number_input("Semente aleatória", value=42, step=1)
        st.session_state["random_seed"] = seed

# ── Página principal ───────────────────────────────────────────────────────────
st.title("⚽ CopIA — Copa do Mundo 2026 com Inteligência Artificial")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Seleções Analisadas", "48")
with col2:
    st.metric("Simulações (default)", "100.000")
with col3:
    st.metric("Features por Jogo", "47+")
with col4:
    st.metric("Modelos no Ensemble", "3 + stacking")

st.markdown("---")

st.markdown("""
## O que é o CopIA?

O **CopIA** é um projeto de Inteligência Artificial aplicado à Copa do Mundo 2026.
Ele combina técnicas modernas de ML, simulação probabilística e Reinforcement Learning
para responder perguntas como:

- 🏆 Qual seleção tem maior probabilidade de ser campeã?
- 📈 O que os dados dizem sobre as chances do Brasil?
- 🎲 Em quantas das 100.000 simulações a Argentina chega à final?
- 🤖 O que um agente de RL aprende sobre decisões táticas de substituição?

## Como usar

Navegue pelas páginas no menu lateral para explorar cada módulo do projeto:

| Página | O que você vai encontrar |
|---|---|
| **Probabilidades** | Ranking de times por P(campeão) com intervalos de confiança |
| **Bracket** | Chaveamento interativo com probabilidades por fase |
| **Explainability** | SHAP values explicando cada predição em linguagem natural |
| **Monte Carlo** | Execute simulações ao vivo e explore cenários "e se?" |
| **Agente RL** | Veja um agente treinado tomando decisões táticas em tempo real |
| **Chat** | Converse com o CopIA sobre qualquer aspecto da Copa |

## Fase 1 — Disponível Agora
""")

# ── Cards de status ────────────────────────────────────────────────────────────
col_a, col_b, col_c = st.columns(3)

with col_a:
    st.success("""
    **✅ Fase 1 — Previsão**

    Pipeline de ML completo com dados históricos de 50+ anos,
    ensemble de modelos, Monte Carlo com 100k simulações e
    explicabilidade SHAP.
    """)

with col_b:
    st.info("""
    **🔄 Fase 2 — Agente RL**

    Ambiente Gymnasium customizado para treinamento de agente
    tático. Q-Learning, DQN e PPO implementados.
    Em desenvolvimento ativo.
    """)

with col_c:
    st.warning("""
    **📋 Em breve — GenAI**

    Agente conversacional com RAG sobre 90 anos de Copas,
    narração de highlights e análise tática em linguagem natural.
    """)

st.markdown("---")
st.caption("CopIA v0.1.0 · Desenvolvido por Ahirton Lopes · MIT License · "
           "[GitHub](https://github.com/ahirtonlopes/copa-ia)")
