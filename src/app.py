"""Interactive set explorer (Streamlit).

App allows users to pick a set, see its energy arc and the vibe journey underneath it, then play with the recommender
for song suggestions.

Everything runs off the artifacts the earlier stages already produced
(data/processed/vibes.parquet and outputs/lstm.pt), so it loads instantly and
needs no audio.

Run:
    streamlit run src/app.py
"""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from src.recommend import load_context, predict_next_energy, recommend_next

# A consistent colour per vibe (by cluster index), tuned for a dark background.
VIBE_COLORS = ["#38bdf8", "#fb923c", "#4ade80", "#f87171", "#c084fc"]

st.set_page_config(page_title="DJ Set Energy Arc Modeller", page_icon="🎛️",
                   layout="wide")

# --- styling -----------------------------------------------------------------
st.markdown(
    """
    <style>
      .stApp {
        background: radial-gradient(1200px 600px at 20% -10%, #1b1140 0%, #0a0a12 55%);
        color: #e8e8f0;
      }
      .hero {
        padding: 1.6rem 2rem; border-radius: 18px; margin-bottom: 1.2rem;
        background: linear-gradient(120deg, #7c3aed 0%, #db2777 60%, #f59e0b 100%);
        box-shadow: 0 10px 40px rgba(124,58,237,0.35);
      }
      .hero h1 { margin: 0; font-size: 2.1rem; color: #fff; letter-spacing: -0.5px; }
      .hero p  { margin: 0.35rem 0 0; color: rgba(255,255,255,0.9); font-size: 1rem; }
      .metric-card {
        background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px; padding: 1rem 1.2rem;
      }
      .metric-card .label { font-size: 0.78rem; color: #9aa0b5; text-transform: uppercase;
        letter-spacing: 1px; }
      .metric-card .value { font-size: 1.5rem; font-weight: 700; margin-top: 0.2rem; }
      .rec-card {
        background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);
        border-left: 5px solid var(--accent); border-radius: 12px;
        padding: 0.9rem 1.1rem; margin-bottom: 0.7rem;
      }
      .rec-card .rank { font-size: 0.75rem; color: #9aa0b5; letter-spacing: 1px; }
      .rec-card .name { font-size: 1.15rem; font-weight: 700; margin: 0.1rem 0 0.5rem; }
      .bar-track { background: rgba(255,255,255,0.08); border-radius: 6px; height: 8px; }
      .bar-fill { background: var(--accent); height: 8px; border-radius: 6px; }
      .sub { color: #9aa0b5; font-size: 0.85rem; margin-top: 0.4rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_context():
    return load_context()


def metric_card(col, label, value):
    col.markdown(
        f"<div class='metric-card'><div class='label'>{label}</div>"
        f"<div class='value'>{value}</div></div>",
        unsafe_allow_html=True,
    )


def energy_arc_figure(sub):
    minutes = sub["t_start_sec"] / 60.0
    fig = go.Figure()
    # Raw energy, faint, for texture.
    fig.add_trace(go.Scatter(x=minutes, y=sub["energy_norm"], mode="lines",
                             line=dict(color="rgba(255,255,255,0.18)", width=1),
                             name="raw energy", hoverinfo="skip"))
    # Smoothed arc as the headline line.
    fig.add_trace(go.Scatter(x=minutes, y=sub["energy_smooth"], mode="lines",
                             line=dict(color="#f5f5ff", width=2.5), name="arc",
                             hoverinfo="skip"))
    # Markers coloured by vibe tie the arc to the vibe journey.
    fig.add_trace(go.Scatter(
        x=minutes, y=sub["energy_smooth"], mode="markers",
        marker=dict(size=7, color=[VIBE_COLORS[v % len(VIBE_COLORS)] for v in sub["vibe"]]),
        text=sub["vibe_label"], name="vibe",
        hovertemplate="%{x:.0f} min<br>energy %{y:.2f}<br>%{text}<extra></extra>"))
    fig.update_layout(
        height=360, margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#cfd2e0", showlegend=False,
        xaxis=dict(title="minutes into set", gridcolor="rgba(255,255,255,0.06)"),
        yaxis=dict(title="relative energy", range=[0, 1],
                   gridcolor="rgba(255,255,255,0.06)"))
    return fig


def vibe_journey_figure(sub):
    minutes = sub["t_start_sec"] / 60.0
    fig = go.Figure(go.Scatter(
        x=minutes, y=[1] * len(sub), mode="markers",
        marker=dict(size=18, symbol="square",
                    color=[VIBE_COLORS[v % len(VIBE_COLORS)] for v in sub["vibe"]]),
        text=sub["vibe_label"],
        hovertemplate="%{x:.0f} min<br>%{text}<extra></extra>"))
    fig.update_layout(
        height=110, margin=dict(l=10, r=10, t=10, b=30),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#cfd2e0",
        xaxis=dict(title="minutes into set", gridcolor="rgba(255,255,255,0.06)"),
        yaxis=dict(visible=False, range=[0.5, 1.5]))
    return fig


def rec_card(rank, row, max_score):
    accent = VIBE_COLORS[int(row["vibe"]) % len(VIBE_COLORS)]
    width = int(100 * row["score"] / max_score) if max_score else 0
    st.markdown(
        f"""
        <div class='rec-card' style='--accent:{accent}'>
          <div class='rank'>#{rank}</div>
          <div class='name'>{row['vibe_label']}</div>
          <div class='bar-track'><div class='bar-fill' style='width:{width}%'></div></div>
          <div class='sub'>fit score {row['score']:.2f} &nbsp;·&nbsp;
            usually follows {int(row['P(follows current)'] * 100)}% of the time
            &nbsp;·&nbsp; typical energy {row['vibe_energy']:.2f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    ctx = get_context()
    df = ctx["df"]

    st.markdown(
        "<div class='hero'><h1>🎛️ DJ Set Energy Arc Modeller</h1>"
        "<p>The anatomy of a set: read its energy arc, trace its vibes, and plan "
        "the next move.</p></div>",
        unsafe_allow_html=True,
    )

    set_ids = sorted(df["set_id"].unique())
    set_id = st.sidebar.selectbox("Choose a set", set_ids)
    sub = df[df["set_id"] == set_id].sort_values("window_index").reset_index(drop=True)

    # --- headline metrics ---
    c1, c2, c3, c4 = st.columns(4)
    metric_card(c1, "Set length", f"{sub['t_end_sec'].max() / 60:.0f} min")
    metric_card(c2, "Windows", f"{len(sub)}")
    metric_card(c3, "Dominant vibe", sub["vibe_label"].mode().iloc[0])
    metric_card(c4, "Average energy", f"{sub['energy_norm'].mean():.2f}")

    st.subheader("Energy arc")
    st.plotly_chart(energy_arc_figure(sub), width='stretch')
    st.subheader("Vibe journey")
    st.plotly_chart(vibe_journey_figure(sub), width='stretch')

    # --- recommender ---
    st.subheader("Plan the next move")
    left, right = st.columns([1, 1.3])
    with left:
        max_min = float(sub["t_start_sec"].max() / 60.0)
        where = st.slider("Pretend I'm this far into the set (minutes)",
                          0.0, max_min, max_min / 2, step=0.5)
        target = st.slider("Target energy for what comes next", 0.0, 1.0, 0.8, 0.05)
        recent = sub[sub["t_start_sec"] / 60.0 <= where]
        if recent.empty:
            recent = sub.iloc[:1]
        current_label = ctx["labels"][int(recent["vibe"].iloc[-1])]
        natural = predict_next_energy(recent, ctx)

        st.markdown(f"**Right now:** {current_label}")
        if natural is not None:
            st.markdown(f"**LSTM momentum:** natural next energy ≈ `{natural:.2f}`")
        else:
            st.markdown("**LSTM momentum:** not enough history yet")

    with right:
        st.markdown("**Suggested next vibes**")
        recs = recommend_next(recent, target, ctx)
        max_score = recs["score"].max()
        for i, (_, row) in enumerate(recs.iterrows(), start=1):
            rec_card(i, row, max_score)

    st.caption("Built from pure-DSP librosa features, an LSTM energy model, and "
               "vibe clusters. No audio or API keys needed at runtime.")


if __name__ == "__main__":
    main()
