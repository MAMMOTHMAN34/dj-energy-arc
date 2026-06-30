"""Set-planning recommender.

Given the last few windows of a set and a target energy, it aims to find what vibe comes next.

It blends these two ideas:

  - From the real sets I build a transition matrix to find out what vibe tends to
    follow what. DJs, in general, do not jump from a dark low-energy intro straight to a
    high BPM track, and the matrix captures that habit so suggestions stay
    realistic.
  - Each vibe has a typical energy. To steer towards the target the user
    asked for, I score candidate vibes by how close their energy sits to it.

The trained LSTM rides along too: it reports the natural next energy implied by
the recent windows (the set's momentum), so I can see whether the target asks the
set to push harder or ease off relative to where it was already heading.

Outputs:
  - figures/vibe_transitions.png   the vibe-to-vibe transition matrix
  - a worked demo printed to the console

Run:
    python -m src.recommend
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.train_lstm import FEATURES, EnergyLSTM
from src.utils import load_config, resolve


def build_transition_matrix(df: pd.DataFrame, n_vibes: int) -> np.ndarray:
    """P(next vibe | current vibe), counted from consecutive windows in each set."""
    M = np.zeros((n_vibes, n_vibes))
    for _, sub in df.groupby("set_id"):
        seq = sub.sort_values("window_index")["vibe"].to_numpy()
        for a, b in zip(seq[:-1], seq[1:]):
            M[a, b] += 1
    row_sums = M.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0   # avoid divide-by-zero for an unseen vibe
    return M / row_sums


def load_context() -> dict:
    """Assemble everything the recommender needs: vibe stats, transitions, LSTM."""
    cfg = load_config()
    processed_dir = resolve(cfg["paths"]["processed"])
    df = pd.read_parquet(processed_dir / "vibes.parquet")
    n_vibes = cfg["cluster"]["n_vibes"]

    labels = df.groupby("vibe")["vibe_label"].first().to_dict()
    vibe_energy = df.groupby("vibe")["energy_norm"].mean()
    T = build_transition_matrix(df, n_vibes)

    # Scaler for the LSTM. I rebuild it from the full feature table here; this is a
    # planning tool, not a held-out evaluation, so using all rows is fine.
    mean, std = df[FEATURES].mean(), df[FEATURES].std().replace(0, 1.0)

    model = EnergyLSTM(len(FEATURES), cfg["model"]["hidden_size"])
    model.load_state_dict(torch.load(resolve("outputs") / "lstm.pt"))
    model.eval()

    return {
        "df": df, "labels": labels, "vibe_energy": vibe_energy, "T": T,
        "mean": mean, "std": std, "model": model,
        "lookback": cfg["model"]["lookback"],
        "w_t": cfg["recommend"]["transition_weight"],
        "w_e": cfg["recommend"]["energy_weight"],
    }


def predict_next_energy(recent: pd.DataFrame, ctx: dict):
    """The LSTM's view of the natural next energy, given the recent windows."""
    sub = recent.tail(ctx["lookback"])
    if len(sub) < ctx["lookback"]:
        return None
    x = ((sub[FEATURES] - ctx["mean"]) / ctx["std"]).to_numpy(np.float32)[None, ...]
    with torch.no_grad():
        return float(ctx["model"](torch.tensor(x)).item())


def recommend_next(recent: pd.DataFrame, target_energy: float, ctx: dict, k: int = 3):
    """Rank the next vibe to play, blending realistic transitions with the target."""
    current = int(recent["vibe"].iloc[-1])
    trans = ctx["T"][current]                         # realism term, already 0-1
    energy = ctx["vibe_energy"].to_numpy()
    energy_close = np.clip(1 - np.abs(energy - target_energy), 0, 1)  # intent term

    # Blend the raw transition probability (how usual the move is) with how well
    # the candidate's energy matches the target. Using the raw probability rather
    # than rescaling it stops one dominant transition from drowning out intent.
    score = ctx["w_t"] * trans + ctx["w_e"] * energy_close

    order = np.argsort(score)[::-1][:k]
    return pd.DataFrame(
        {
            "vibe": order,
            "vibe_label": [ctx["labels"][v] for v in order],
            "P(follows current)": [round(trans[v], 2) for v in order],
            "vibe_energy": [round(energy[v], 2) for v in order],
            "score": [round(score[v], 3) for v in order],
        }
    )


def _plot_transitions(ctx, out_path):
    T, labels = ctx["T"], ctx["labels"]
    names = [labels[i] for i in range(len(T))]
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(T, cmap="magma", vmin=0, vmax=T.max())
    ax.set_xticks(range(len(T)), names, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(T)), names, fontsize=8)
    for i in range(len(T)):
        for j in range(len(T)):
            ax.text(j, i, f"{T[i, j]:.2f}", ha="center", va="center",
                    color="white" if T[i, j] < T.max() * 0.6 else "black", fontsize=8)
    ax.set_xlabel("next vibe")
    ax.set_ylabel("current vibe")
    ax.set_title("How vibes follow one another")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _demo(ctx) -> None:
    """A worked example: drop into the middle of a set and ask for advice."""
    df = ctx["df"]
    set_id = sorted(df["set_id"].unique())[0]
    sub = df[df["set_id"] == set_id].sort_values("window_index")
    cut = len(sub) // 2
    recent = sub.iloc[:cut]
    current_label = ctx["labels"][int(recent["vibe"].iloc[-1])]

    natural = predict_next_energy(recent, ctx)
    print(f"Demo set: {set_id}")
    print(f"  {cut} windows in, currently in '{current_label}' "
          f"(energy {recent['energy_norm'].iloc[-1]:.2f})")
    if natural is not None:
        print(f"  LSTM says the natural next energy is ~{natural:.2f}")

    for intent, target in [("lift to peak", 0.85), ("cool it down", 0.30)]:
        print(f"\n  If I want to {intent} (target energy {target}):")
        recs = recommend_next(recent, target, ctx)
        print(recs.to_string(index=False))


def main() -> None:
    ctx = load_context()
    _plot_transitions(ctx, resolve("figures") / "vibe_transitions.png")
    _demo(ctx)
    print(f"\nSaved transition figure to {resolve('figures') / 'vibe_transitions.png'}")


if __name__ == "__main__":
    main()
