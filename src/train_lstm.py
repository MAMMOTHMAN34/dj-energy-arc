"""Model the energy arc with an LSTM.

Given the last few windows of a set, it predicts the energy of the next window and learns how sets rise and fall.

Two decisions I made:

  - I held out whole sets for validation. Random windows
    would let the model peek at a set it is also being tested on, which inflates
    the score. It was better to judge it on unseen sets.
  - I compared against a persistence baseline (predict "next energy = current
    energy"). For smooth time series that baseline is surprisingly hard to beat,
    so beating it is what proves the LSTM learnt the shape.

Each input step carries a few features: energy, tempo, brightness,
danceability and position.

Outputs:
  - outputs/lstm.pt              the trained weights (gitignored, re-trainable)
  - figures/lstm_loss.png        train vs validation loss per epoch
  - figures/lstm_prediction.png  predicted vs actual arc on a held-out set

Run:
    python -m src.train_lstm
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.utils import load_config, resolve

# What each timestep feeds the model. energy_norm is the autoregressive signal;
# the rest give it context for where the energy is likely to head next.
FEATURES = ["energy_norm", "bpm", "spectral_centroid", "danceability", "zcr", "t_norm"]
TARGET = "energy_norm"


class EnergyLSTM(nn.Module):
    """A small LSTM that maps a window of features to the next energy value."""

    def __init__(self, n_features: int, hidden_size: int):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)          # out: (batch, steps, hidden)
        return self.head(out[:, -1, :]).squeeze(-1)  # use the last step only


def split_sets(set_ids, val_fraction, seed):
    """Pick whole sets for validation, deterministically."""
    rng = np.random.default_rng(seed)
    shuffled = list(set_ids)
    rng.shuffle(shuffled)
    n_val = max(1, round(val_fraction * len(shuffled)))
    return shuffled[n_val:], shuffled[:n_val]  # train, val


def make_sequences(df, set_ids, lookback, scaler):
    """Turn each set into (lookback -> next-energy) samples.

    Sequences never cross a set boundary. For every sample I also keep the
    persistence prediction (the most recent observed energy) and a little
    metadata so I can plot a real set's arc later."""
    Xs, ys, persist, meta = [], [], [], []
    for sid in set_ids:
        sub = df[df["set_id"] == sid].sort_values("window_index")
        feats = ((sub[FEATURES] - scaler["mean"]) / scaler["std"]).to_numpy(np.float32)
        target = sub[TARGET].to_numpy(np.float32)
        energy = sub["energy_norm"].to_numpy(np.float32)
        minutes = (sub["t_start_sec"].to_numpy() / 60.0)
        for i in range(len(sub) - lookback):
            Xs.append(feats[i : i + lookback])
            ys.append(target[i + lookback])
            persist.append(energy[i + lookback - 1])   # "next = current"
            meta.append((sid, minutes[i + lookback]))
    return (
        np.asarray(Xs, np.float32),
        np.asarray(ys, np.float32),
        np.asarray(persist, np.float32),
        meta,
    )


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a, b):
    return float(np.mean(np.abs(a - b)))


def train() -> None:
    cfg = load_config()
    m = cfg["model"]
    seed = cfg["cluster"]["random_state"]  # reuse the one project-wide seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    processed_dir = resolve(cfg["paths"]["processed"])
    out_dir = resolve("outputs")
    fig_dir = resolve("figures")

    df = pd.read_parquet(processed_dir / "arcs.parquet")
    train_ids, val_ids = split_sets(df["set_id"].unique(), m["val_fraction"], seed)
    print(f"Train on {len(train_ids)} sets, validate on {len(val_ids)}: {val_ids}")

    # Scale features on TRAINING rows only, so the val sets leak nothing.
    train_rows = df[df["set_id"].isin(train_ids)]
    scaler = {
        "mean": train_rows[FEATURES].mean(),
        "std": train_rows[FEATURES].std().replace(0, 1.0),
    }

    lookback = m["lookback"]
    Xtr, ytr, _, _ = make_sequences(df, train_ids, lookback, scaler)
    Xva, yva, persist_va, meta_va = make_sequences(df, val_ids, lookback, scaler)
    print(f"{len(Xtr)} training sequences, {len(Xva)} validation sequences")

    loader = DataLoader(
        TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
        batch_size=m["batch_size"],
        shuffle=True,
    )

    model = EnergyLSTM(len(FEATURES), m["hidden_size"])
    opt = torch.optim.Adam(model.parameters(), lr=m["learning_rate"])
    loss_fn = nn.MSELoss()

    Xva_t, yva_t = torch.from_numpy(Xva), torch.from_numpy(yva)
    train_losses, val_losses = [], []

    for epoch in range(m["epochs"]):
        model.train()
        batch_losses = []
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            batch_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_pred = model(Xva_t)
            val_loss = loss_fn(val_pred, yva_t).item()
        train_losses.append(float(np.mean(batch_losses)))
        val_losses.append(val_loss)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"epoch {epoch + 1:>2}: train {train_losses[-1]:.4f} "
                  f"val {val_loss:.4f}")

    # Final scores against the persistence baseline.
    model.eval()
    with torch.no_grad():
        pred_va = model(Xva_t).numpy()

    print("\nValidation (held-out sets):")
    print(f"  persistence  RMSE {rmse(persist_va, yva):.4f}  MAE {mae(persist_va, yva):.4f}")
    print(f"  LSTM         RMSE {rmse(pred_va, yva):.4f}  MAE {mae(pred_va, yva):.4f}")
    gain = 100 * (1 - rmse(pred_va, yva) / rmse(persist_va, yva))
    verdict = "beats" if gain > 0 else "does NOT beat"
    print(f"  -> LSTM {verdict} the baseline by {gain:+.1f}% on RMSE")

    torch.save(model.state_dict(), out_dir / "lstm.pt")

    _plot_loss(train_losses, val_losses, fig_dir / "lstm_loss.png")
    _plot_prediction(meta_va, yva, pred_va, persist_va, val_ids[0],
                     fig_dir / "lstm_prediction.png")
    print(f"\nSaved model to {out_dir / 'lstm.pt'} and figures to {fig_dir}")


def _plot_loss(train_losses, val_losses, out_path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(train_losses, label="train")
    ax.plot(val_losses, label="validation")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title("LSTM training")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _plot_prediction(meta, actual, pred, persist, set_id, out_path):
    """Plot predicted vs actual energy across one held-out set."""
    idx = [i for i, (sid, _) in enumerate(meta) if sid == set_id]
    minutes = [meta[i][1] for i in idx]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(minutes, [actual[i] for i in idx], label="actual", linewidth=2)
    ax.plot(minutes, [pred[i] for i in idx], label="LSTM", linewidth=1.5)
    ax.plot(minutes, [persist[i] for i in idx], label="persistence",
            linestyle="--", alpha=0.6)
    ax.set_xlabel("minutes into set")
    ax.set_ylabel("relative energy")
    ax.set_title(f"Predicting the arc of a held-out set: {set_id}")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    train()
