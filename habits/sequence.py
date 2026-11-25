"""The professor's GRU and CNN-GRU (Yaprakdal 2024) as habit models.

Same question, same split and same scoring as the other habit models:
"will this appliance be switched on in this hour?", trained on the first 60%
of a home's hours, threshold chosen on the next 20%, scored on the last 20%.

The difference is the input. The random forest gets 15 hand-made features
(habits/features.py); the networks get the raw past week, hour by hour, and
learn their own patterns. For the hour h being predicted, the window is the
WINDOW hours before it on a full calendar grid; at each step:

    history   the appliance's own use (0/1), switch-ons of the house's other
              appliances (log 1+n), whether that hour has data at all
    calendar  hour of day and day of week (sine/cosine) of the NEXT step, so
              the last step carries the calendar of hour h itself

Nothing from hour h or later is used, as with the hand-made features.

The networks are energy/deep.py's GRU (50 -> 25 units) and CNN-GRU, unchanged,
with one output unit read as a logit. Training: Adam (1e-3), binary
cross-entropy with positives weighted by negatives/positives (the forest's
"balanced" classes), batches of BATCH, the learning rate halved after 2 flat
epochs, early stopping on the validation weeks' loss after PATIENCE epochs
(best weights restored). The losses at the best and at the last epoch are
returned, to tell overfitting (training loss still falling, validation loss
rising) from an optimiser that is stuck (both flat).
"""

import numpy as np
import pandas as pd
import torch
from torch import nn

from energy.deep import DEVICE, CNNGRU, GRUNet
from habits.models import best_threshold

WINDOW = 168  # one week
N_FEATURES = 7
MAX_EPOCHS = 30
PATIENCE = 5
BATCH = 256
NETWORKS = ["GRU", "CNN-GRU"]


def grid_inputs(labels: pd.Series, other_starts: pd.Series | None) -> tuple[np.ndarray, np.ndarray]:
    """Per-hour inputs on a full calendar grid, padded with WINDOW empty hours in front,
    and each covered hour's position in that padded grid."""
    grid = pd.date_range(labels.index.min(), labels.index.max(), freq="h")
    own = labels.reindex(grid, fill_value=0).to_numpy(float)
    covered = pd.Series(1.0, index=labels.index).reindex(grid, fill_value=0.0).to_numpy()
    others = (other_starts.reindex(grid, fill_value=0).to_numpy(float) if other_starts is not None
              else np.zeros(len(grid)))
    # calendar of the next hour (one hour past the grid's end for the last step)
    nxt = grid + pd.Timedelta(hours=1)
    hour, dow = nxt.hour.to_numpy(), nxt.dayofweek.to_numpy()
    x = np.column_stack([own, np.log1p(others), covered,
                         np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
                         np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)]).astype(np.float32)
    x = np.vstack([np.zeros((WINDOW, N_FEATURES), np.float32), x])
    position = grid.get_indexer(labels.index) + WINDOW  # row of hour h; its window is rows [h-WINDOW, h)
    return x, position


def make(name: str) -> nn.Module:
    if name == "GRU":
        return GRUNet(N_FEATURES)
    if name == "CNN-GRU":
        return CNNGRU(N_FEATURES, steps=WINDOW)
    raise ValueError(name)


def fit_and_predict_sequence(name: str, task, seed: int = 0) -> tuple[np.ndarray, dict]:
    """Like habits.models.fit_and_predict, for the networks."""
    n_test = len(task.test)
    if task.train.nunique() < 2:
        return np.zeros(n_test, dtype=int), {}
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    x, position = grid_inputs(task.labels, task.other_starts)
    x = torch.from_numpy(x).to(DEVICE)
    steps = torch.arange(-WINDOW, 0, device=DEVICE)
    a, b = len(task.train), len(task.train) + len(task.val)
    pos = torch.from_numpy(position).to(DEVICE)
    y = torch.from_numpy(task.labels.to_numpy(np.float32)).to(DEVICE)

    def windows(rows):  # (n,) positions -> (n, WINDOW, N_FEATURES)
        return x[rows[:, None] + steps[None, :]]

    def scores(model, part):
        with torch.no_grad():
            return torch.cat([torch.sigmoid(model(windows(part[i:i + 4096])))
                              for i in range(0, len(part), 4096)]).cpu().numpy()

    train, val, test = pos[:a], pos[a:b], pos[b:]
    y_train, y_val = y[:a], y[a:b]
    positives = float(y_train.sum())
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((a - positives) / positives, device=DEVICE))
    model = make(name).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    best, best_state, waited, epochs = np.inf, None, 0, 0
    curve = {}  # losses at the best epoch and at the last one: overfitting, or a stuck optimiser?
    for epochs in range(1, MAX_EPOCHS + 1):
        model.train()
        order = torch.from_numpy(rng.permutation(a)).to(DEVICE)
        train_loss = 0.0
        for i in range(0, a, BATCH):
            batch = order[i:i + BATCH]
            opt.zero_grad()
            loss = loss_fn(model(windows(train[batch])), y_train[batch])
            loss.backward()
            opt.step()
            train_loss += loss.item() * len(batch) / a
        model.eval()
        with torch.no_grad():
            val_loss = float(np.mean([loss_fn(model(windows(val[i:i + 4096])), y_val[i:i + 4096]).item()
                                      for i in range(0, len(val), 4096)]))
        plateau.step(val_loss)
        curve.update(train_loss_last=train_loss, val_loss_last=val_loss, lr_last=opt.param_groups[0]["lr"])
        if val_loss < best - 1e-5:
            best, waited = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            curve.update(best_epoch=epochs, train_loss_best=train_loss, val_loss_best=val_loss)
        else:
            waited += 1
            if waited >= PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    threshold = best_threshold(task.val, scores(model, val))
    return (scores(model, test) >= threshold).astype(int), {"threshold": threshold, "epochs": epochs, **curve}
