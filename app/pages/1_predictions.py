"""
CopIA — Página de Probabilidades de Campeão
Mostra as probabilidades de título, análise por confederação
e comparação entre modelos.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.models.baseline import PoissonModel
from src.simulation.monte_carlo import TournamentSimulator, COPA_2026_GROUPS

st.set_page_config(page_title="Probabilidades — CopIA", page_icon="🏆", layout="wide")
st.title("🏆 Probabilidades de Campeão — Copa 2026")
st.caption("Baseado em simulação Monte Carlo com modelo Poisson + Dixon-Coles")

# ── Configurações ──────────────────────────────────────────────────────────────
n_sims = st.session_state.get("n_simulations", 10_000)
seed = st.session_state.get("random_seed", 42)

col_run, col_info = st.columns([1, 3])
with col_run:
    run_button = st.button("▶️ Rodar Simulação", type="primary", use_container_width=True)
with col_info:
    st.info(f"Configurado para **{n_sims:,} simulações**. "
            "Ajuste na barra lateral da página inicial.")

# ── Simulação ──────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner="Simulando o torneio...")
def run_simulation(n: int, seed: int):
    model = PoissonModel()
    simulator = TournamentSimulator(model=model, random_seed=seed)
    return simulator.run(n_simulations=n)


if run_button or "sim_results" not in st.session_state:
    with st.spinner(f"Rodando {n_sims:,} simulações..."):
        results = run_simulation(n_sims, seed)
        st.session_state["sim_results"] = results

results = st.session_state.get("sim_results")

if results is None:
    st.warning("Clique em 'Rodar Simulação' para ver os resultados.")
    st.stop()

# ── Dados ──────────────────────────────────────────────────────────────────────
champ_df = results.get_champion_probabilities(top_n=48)
full_df = results.get_full_probabilities()

# Adiciona confederação
confederation_map = {
    team: group
    for group, teams in COPA_2026_GROUPS.items()
    for team in teams
}
champ_df["group"] = champ_df["team"].map(confederation_map).fillna("?")

# ── KPIs ───────────────────────────────────────────────────────────────────────
st.markdown("### 📊 Resultados da Simulação")
top3 = champ_df.head(3)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(
        "🥇 Favorito",
        top3.iloc[0]["team"],
        f"{top3.iloc[0]['p_champion_pct']:.1f}%",
    )
with col2:
    st.metric(
        "🥈 2º favorito",
        top3.iloc[1]["team"],
        f"{top3.iloc[1]['p_champion_pct']:.1f}%",
    )
with col3:
    st.metric(
        "🥉 3º favorito",
        top3.iloc[2]["team"],
        f"{top3.iloc[2]['p_champion_pct']:.1f}%",
    )
with col4:
    brazil_row = champ_df[champ_df["team"] == "Brasil"]
    brazil_pct = brazil_row["p_champion_pct"].values[0] if len(brazil_row) > 0 else 0
    brazil_pos = champ_df[champ_df["team"] == "Brasil"].index[0] + 1 if len(brazil_row) > 0 else "-"
    st.metric("🇧🇷 Brasil", f"{brazil_pct:.1f}%", f"#{brazil_pos} favorito")

st.markdown("---")

# ── Gráfico principal ──────────────────────────────────────────────────────────
top_n = st.slider("Mostrar top N seleções", 5, 48, 20)
plot_df = champ_df.head(top_n).sort_values("p_champion_pct")

fig = px.bar(
    plot_df,
    x="p_champion_pct",
    y="team",
    orientation="h",
    title=f"Probabilidade de ser campeão — Top {top_n} seleções",
    labels={"p_champion_pct": "Probabilidade (%)", "team": "Seleção"},
    color="p_champion_pct",
    color_continuous_scale="RdYlGn",
    text="p_champion_pct",
)
fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
fig.update_layout(
    height=max(400, top_n * 28),
    showlegend=False,
    coloraxis_showscale=False,
    yaxis={"categoryorder": "total ascending"},
)
st.plotly_chart(fig, use_container_width=True)

# ── Tabela completa de probabilidades por fase ─────────────────────────────────
st.markdown("### 📋 Probabilidades por Fase")

display_df = full_df.copy()
prob_cols = [c for c in display_df.columns if c.startswith("p_") and c != "p_champion"]
all_prob_cols = ["p_champion"] + prob_cols

for col in all_prob_cols:
    if col in display_df.columns:
        display_df[col] = (display_df[col] * 100).round(1).astype(str) + "%"

st.dataframe(
    display_df[["team"] + all_prob_cols].head(48),
    use_container_width=True,
    hide_index=True,
)

# ── Análise por grupo ──────────────────────────────────────────────────────────
st.markdown("### 🔢 Distribuição por Grupo")
group_data = []
for group, teams in COPA_2026_GROUPS.items():
    for team in teams:
        row = champ_df[champ_df["team"] == team]
        p = row["p_champion_pct"].values[0] if len(row) > 0 else 0
        group_data.append({"group": f"Grupo {group}", "team": team, "p_champion": p})

group_df = pd.DataFrame(group_data)
fig2 = px.bar(
    group_df,
    x="team",
    y="p_champion",
    color="group",
    title="Probabilidade de título por grupo",
    labels={"p_champion": "P(Campeão) %", "team": ""},
    barmode="group",
)
fig2.update_layout(xaxis_tickangle=-45, height=450)
st.plotly_chart(fig2, use_container_width=True)

st.caption(
    f"Resultados baseados em {n_sims:,} simulações Monte Carlo · "
    "Modelo: Poisson Bivariado com correção Dixon-Coles · "
    "CopIA v0.1.0"
)
