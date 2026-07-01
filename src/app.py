"""Interactive set explorer and live set planner (Streamlit).

Three tabs:
  - Explore sets: pick any set (built-in or one you uploaded), read its energy
    arc and vibe journey, and ask the recommender what to play next.
  - Live set builder: build a set as you go. Tap the vibes you have played and
    it keeps a running arc and suggests the next vibe to hit your target energy.
  - Add a set: upload your own audio and it is analysed with the same pipeline.

The first two tabs run off the artifacts the batch pipeline already produced
(data/processed/vibes.parquet, outputs/lstm.pt) so they load instantly. Only the
upload tab pulls in librosa, and only when a file is actually dropped.

Run:
    streamlit run src/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# When launched as `streamlit run src/app.py`, only the src/ folder lands on the
# path, so `import src...` fails. Put the repo root on the path first.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.recommend import load_context, predict_next_energy, recommend_next
from src.train_lstm import FEATURES

# A consistent colour per vibe (by cluster index), tuned for a dark background.
VIBE_COLORS = ["#38bdf8", "#fb923c", "#4ade80", "#f87171", "#c084fc"]

st.set_page_config(page_title="DJ Set Energy Arc Modeller", layout="wide")

# --- styling -----------------------------------------------------------------
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;800&family=Inter:wght@400;500&display=swap');

      html, body, .stApp, [class*="css"] { font-family: 'Inter', sans-serif; }
      h1, h2, h3, .hero h1 { font-family: 'Sora', sans-serif; }

      /* Slowly drifting dark gradient background. */
      .stApp {
        background: linear-gradient(135deg, #0a0a12, #160d2e, #0a0a12, #1b0a26);
        background-size: 400% 400%;
        animation: bgshift 22s ease infinite;
        color: #e8e8f0;
      }
      @keyframes bgshift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
      }
      [data-testid="stHeader"] { background: transparent; }

      /* Hero banner with a moving gradient and a glow. */
      .hero {
        position: relative; overflow: hidden;
        padding: 1.8rem 2rem; border-radius: 20px; margin-bottom: 1.4rem;
        background: linear-gradient(120deg, #7c3aed, #db2777, #f59e0b, #7c3aed);
        background-size: 300% 300%; animation: bgshift 12s ease infinite;
        box-shadow: 0 14px 50px rgba(124,58,237,0.4);
      }
      .hero h1 { margin: 0; font-size: 2.3rem; color: #fff; letter-spacing: -0.6px; font-weight: 800; }
      .hero p  { margin: 0.4rem 0 0; color: rgba(255,255,255,0.92); font-size: 1.02rem; }

      /* Animated equaliser bars in the hero corner. */
      .eq { position: absolute; top: 1.6rem; right: 2rem; display: flex; gap: 4px; align-items: flex-end; height: 46px; }
      .eq span { width: 6px; height: 30%; background: rgba(255,255,255,0.9); border-radius: 3px; animation: eq 0.9s ease-in-out infinite; }
      @keyframes eq { 0%, 100% { height: 18%; } 50% { height: 100%; } }

      /* Glassmorphism cards. */
      .metric-card, .rec-card {
        background: rgba(255,255,255,0.05); backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.10); border-radius: 16px;
        padding: 1rem 1.2rem; transition: transform .2s ease, box-shadow .2s ease;
      }
      .metric-card:hover { transform: translateY(-3px); box-shadow: 0 10px 34px rgba(124,58,237,0.30); }
      .metric-card .label { font-size: 0.74rem; color: #9aa0b5; text-transform: uppercase; letter-spacing: 1.2px; }
      .metric-card .value { font-size: 1.5rem; font-weight: 700; margin-top: 0.2rem; font-family: 'Sora', sans-serif; }

      .rec-card { border-left: 5px solid var(--accent); margin-bottom: 0.7rem; }
      .rec-card:hover { box-shadow: 0 0 26px -4px var(--accent); transform: translateY(-2px); }
      .rec-card .rank { font-size: 0.72rem; color: #9aa0b5; letter-spacing: 1px; }
      .rec-card .name { font-size: 1.15rem; font-weight: 700; margin: 0.1rem 0 0.5rem; font-family: 'Sora', sans-serif; }
      .bar-track { background: rgba(255,255,255,0.08); border-radius: 6px; height: 8px; }
      .bar-fill { height: 8px; border-radius: 6px; background: var(--accent); box-shadow: 0 0 12px var(--accent); }
      .sub { color: #9aa0b5; font-size: 0.85rem; margin-top: 0.4rem; }
      .now-chip { display:inline-block; padding: 0.25rem 0.8rem; border-radius: 999px;
        background: var(--accent); color: #0a0a12; font-weight: 700; font-size: 0.9rem; }

      /* Tabs. */
      .stTabs [data-baseweb="tab-list"] { gap: 8px; }
      .stTabs [data-baseweb="tab"] { background: rgba(255,255,255,0.04); border-radius: 12px 12px 0 0; padding: 8px 20px; }
      .stTabs [aria-selected="true"] { background: linear-gradient(120deg,#7c3aed,#db2777); color: #fff !important; }

      /* Buttons with a neon hover. */
      .stButton > button {
        background: rgba(255,255,255,0.06); color: #e8e8f0; width: 100%;
        border: 1px solid rgba(255,255,255,0.14); border-radius: 12px;
        padding: 0.55rem 0.7rem; font-weight: 600; transition: all .2s ease;
      }
      .stButton > button:hover { border-color: #c084fc; box-shadow: 0 0 20px rgba(192,132,252,0.45); transform: translateY(-2px); color: #fff; }

      /* Hide the sidebar toggle entirely (we do not use a sidebar). */
      [data-testid="stSidebar"], [data-testid="stSidebarCollapseButton"],
      [data-testid="collapsedControl"] { display: none !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_context():
    ctx = load_context()
    # Average feature vector per vibe: used to fake a "recent window" in the live
    # builder, where the user gives vibes rather than real audio.
    ctx["vibe_feature_means"] = ctx["df"].groupby("vibe")[FEATURES].mean()
    return ctx


def hero():
    bars = "".join(
        f"<span style='animation-delay:{i * 0.08:.2f}s'></span>" for i in range(16)
    )
    st.markdown(
        f"""
        <div class='hero'>
          <div class='eq'>{bars}</div>
          <h1> DJ Set Energy Arc Modeller</h1>
          <p>Read a set's energy arc, trace its vibes, and plan the next move,
             all from the raw audio.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(col, label, value):
    col.markdown(
        f"<div class='metric-card'><div class='label'>{label}</div>"
        f"<div class='value'>{value}</div></div>",
        unsafe_allow_html=True,
    )


def energy_arc_figure(sub, x=None, x_title="minutes into set"):
    x = sub["t_start_sec"] / 60.0 if x is None else x
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=sub["energy_norm"], mode="lines",
                             line=dict(color="rgba(255,255,255,0.18)", width=1),
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=sub["energy_smooth"], mode="lines",
                             line=dict(color="#f5f5ff", width=2.5), hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=x, y=sub["energy_smooth"], mode="markers",
        marker=dict(size=8, color=[VIBE_COLORS[v % len(VIBE_COLORS)] for v in sub["vibe"]]),
        text=sub["vibe_label"],
        hovertemplate="%{text}<br>energy %{y:.2f}<extra></extra>"))
    fig.update_layout(
        height=360, margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#cfd2e0", showlegend=False,
        xaxis=dict(title=x_title, gridcolor="rgba(255,255,255,0.06)"),
        yaxis=dict(title="relative energy", range=[0, 1], gridcolor="rgba(255,255,255,0.06)"))
    return fig


def vibe_journey_figure(sub, x=None, x_title="minutes into set"):
    x = sub["t_start_sec"] / 60.0 if x is None else x
    fig = go.Figure(go.Scatter(
        x=x, y=[1] * len(sub), mode="markers",
        marker=dict(size=18, symbol="square",
                    color=[VIBE_COLORS[v % len(VIBE_COLORS)] for v in sub["vibe"]]),
        text=sub["vibe_label"], hovertemplate="%{text}<extra></extra>"))
    fig.update_layout(
        height=110, margin=dict(l=10, r=10, t=10, b=30),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#cfd2e0",
        xaxis=dict(title=x_title, gridcolor="rgba(255,255,255,0.06)"),
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
          <div class='sub'>fit {row['score']:.2f} &nbsp;·&nbsp;
            follows {int(row['P(follows current)'] * 100)}% of the time
            &nbsp;·&nbsp; typical energy {row['vibe_energy']:.2f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def recommendation_block(recent, ctx, target):
    """Shared LSTM-momentum readout + ranked next-vibe cards."""
    current_label = ctx["labels"][int(recent["vibe"].iloc[-1])]
    natural = predict_next_energy(recent, ctx)
    accent = VIBE_COLORS[int(recent["vibe"].iloc[-1]) % len(VIBE_COLORS)]
    st.markdown(f"Right now: <span class='now-chip' style='--accent:{accent}'>"
                f"{current_label}</span>", unsafe_allow_html=True)
    if natural is not None:
        st.markdown(f"**LSTM momentum** — natural next energy ≈ `{natural:.2f}`")
    else:
        st.caption("Add a few more steps for the LSTM to read the momentum.")
    st.markdown("**Suggested next vibes**")
    recs = recommend_next(recent, target, ctx)
    max_score = recs["score"].max()
    for i, (_, row) in enumerate(recs.iterrows(), start=1):
        rec_card(i, row, max_score)


# --- tab 1: explore ----------------------------------------------------------
def tab_explore(ctx):
    df = ctx["df"]
    base = sorted(df["set_id"].unique())
    uploaded = list(st.session_state.get("uploaded", {}).keys())
    choice = st.selectbox("Choose a set", base + uploaded)

    if choice in st.session_state.get("uploaded", {}):
        sub = st.session_state["uploaded"][choice]
    else:
        sub = df[df["set_id"] == choice].sort_values("window_index").reset_index(drop=True)

    c1, c2, c3, c4 = st.columns(4)
    metric_card(c1, "Set length", f"{sub['t_end_sec'].max() / 60:.0f} min")
    metric_card(c2, "Windows", f"{len(sub)}")
    metric_card(c3, "Dominant vibe", sub["vibe_label"].mode().iloc[0])
    metric_card(c4, "Average energy", f"{sub['energy_norm'].mean():.2f}")

    st.subheader("Energy arc")
    st.plotly_chart(energy_arc_figure(sub), width="stretch")
    st.subheader("Vibe journey")
    st.plotly_chart(vibe_journey_figure(sub), width="stretch")

    st.subheader("Plan the next move")
    left, right = st.columns([1, 1.3])
    with left:
        max_min = float(sub["t_start_sec"].max() / 60.0)
        where = st.slider("Pretend I'm this far in (minutes)", 0.0, max_min,
                          max_min / 2, step=0.5)
        target = st.slider("Target energy for what comes next", 0.0, 1.0, 0.8, 0.05)
        recent = sub[sub["t_start_sec"] / 60.0 <= where]
        if recent.empty:
            recent = sub.iloc[:1]
    with right:
        recommendation_block(recent, ctx, target)


# --- tab 2: live builder -----------------------------------------------------
def live_recent_df(played, ctx):
    """Fake a recent-windows table from the vibes the DJ says they have played."""
    vfm = ctx["vibe_feature_means"]
    n = len(played)
    rows = []
    for i, v in enumerate(played):
        r = vfm.loc[v].to_dict()
        r["vibe"] = v
        r["t_norm"] = i / (n - 1) if n > 1 else 0.0
        r["vibe_label"] = ctx["labels"][v]
        r["energy_smooth"] = r["energy_norm"]
        rows.append(r)
    return pd.DataFrame(rows)


def tab_live(ctx):
    st.markdown("Build a set as you go. Tap each vibe as you play it, and I will "
                "keep the running arc and suggest where to take it next.")
    st.session_state.setdefault("played", [])

    st.markdown("**Add the vibe you just played**")
    cols = st.columns(len(ctx["labels"]))
    for v, col in zip(sorted(ctx["labels"]), cols):
        with col:
            if st.button(ctx["labels"][v], key=f"add_{v}"):
                st.session_state["played"].append(v)

    a, b, _ = st.columns([1, 1, 4])
    if a.button("↶ Undo") and st.session_state["played"]:
        st.session_state["played"].pop()
    if b.button("✕ Reset"):
        st.session_state["played"] = []

    played = st.session_state["played"]
    if not played:
        st.info("No tracks yet. Add the vibe of your opener to get started.")
        return

    recent = live_recent_df(played, ctx)
    st.subheader(f"Your set so far ({len(played)} track{'s' if len(played) > 1 else ''})")
    st.plotly_chart(
        energy_arc_figure(recent, x=list(range(1, len(recent) + 1)),
                          x_title="track number"),
        width="stretch")

    left, right = st.columns([1, 1.3])
    with left:
        target = st.slider("Where do you want to take it next?", 0.0, 1.0, 0.8, 0.05,
                           key="live_target")
    with right:
        recommendation_block(recent, ctx, target)


# --- tab 3: upload -----------------------------------------------------------
def tab_add(ctx):
    st.markdown("Upload your own set (mp3, wav, m4a, flac). It runs through the "
                "exact same librosa pipeline and joins the others under **Explore**.")
    st.caption("Long sets take a little while to analyse, and very long uploads "
               "are best run locally rather than on the hosted demo.")
    up = st.file_uploader("Drop an audio file", type=["mp3", "wav", "m4a", "flac", "ogg"])
    if up is None:
        return

    st.session_state.setdefault("uploaded", {})
    if up.name not in st.session_state["uploaded"]:
        import os
        import tempfile

        from src.analyse import analyse_audio  # lazy: only import librosa on upload

        with st.spinner(f"Analysing {up.name}..."):
            suffix = os.path.splitext(up.name)[1] or ".mp3"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tf:
                tf.write(up.getbuffer())
                tmp = tf.name
            try:
                df = analyse_audio(tmp, set_id=up.name)
            finally:
                os.unlink(tmp)
        st.session_state["uploaded"][up.name] = df
        st.success(f"Added {up.name}. It is now selectable under Explore.")

    sub = st.session_state["uploaded"][up.name]
    st.plotly_chart(energy_arc_figure(sub), width="stretch")
    st.plotly_chart(vibe_journey_figure(sub), width="stretch")


def main():
    ctx = get_context()
    hero()
    explore, live, add = st.tabs(["Explore sets", "Live set builder", "⬆Add a set"])
    with explore:
        tab_explore(ctx)
    with live:
        tab_live(ctx)
    with add:
        tab_add(ctx)
    st.caption("LSTM energy model")


if __name__ == "__main__":
    main()
