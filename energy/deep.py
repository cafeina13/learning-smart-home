"""The professor's deep forecasters (PyTorch): two CNNs, two BiLSTMs, a GRU and the CNN-GRU hybrid.

Yaprakdal (2023), Figs. 2-3. The input is the 144-hour window as a one-channel
"image" (144 time steps x features) for the CNNs and as a sequence for the RNNs.

CNN1   conv 3x3 blocks with 8, 16, 32, 64 filters, each with batch norm and ReLU,
       pooling (2x2, stride 2, over time only once the feature axis is too small
       to halve), dropout 0.2, fully connected 512, output
CNN2   the same with 8, 16, 32, 32 filters, then the output layer
RNN1   bidirectional LSTM, 64 units, fully connected output (the paper does not
       give the unit counts)
RNN2   bidirectional LSTM, 128 units

From Yaprakdal (2024), "Hybrid CNN-GRU model for monthly electricity price
forecasting", Edelweiss Applied Science and Technology 8(6):431-443, Sections 2.2-2.4:

CNN-GRU two branches on the same input, concatenated into one dense output unit:
        CNN branch: 1-D convolutions with 32 and 64 filters, kernel 2, ReLU,
        max pooling 2, flatten; GRU branch: a GRU of 50 units returning
        sequences into a GRU of 25 units
GRU     the GRU branch alone (her paper's GRU benchmark)

Her model was trained with batches of 7 on 60 monthly values and inputs scaled to
[0, 1]; here batches of 64 (thousands of hourly windows per home) and the Energies
paper's preprocessing, so every forecaster sees the same inputs.

Deviations, stated: the paper's figures show the networks emitting a 144-step
output; the task is the next hour's value, so ours emit that one value. The first
fully connected layer of CNN1 (5096 units in the paper) is 512 here: the input
after pooling is small, and a 5096-unit layer on a few thousand training hours
would only memorise them.

Training (for every network, as in the 2024 paper): Adam (learning rate 1e-3),
mean squared error on the z-scored target, batches of 64, up to MAX_EPOCHS epochs;
the learning rate is halved when the validation loss stops improving for 2 epochs;
early stopping (restoring the best weights) after PATIENCE epochs without
improvement. Validation = the last 10% of the training hours, in time order.
"""

import numpy as np
import torch
from torch import nn

# A GPU when there is one (e.g. on Kaggle), the CPU otherwise. Same results either way,
# up to floating-point noise.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_EPOCHS = 50
PATIENCE = 5
BATCH = 64


class CNN(nn.Module):
    def __init__(self, n_features: int, filters: tuple[int, ...], hidden: int | None):
        super().__init__()
        layers, channels, width = [], 1, n_features
        for i, f in enumerate(filters):
            layers += [nn.Conv2d(channels, f, 3, padding=1), nn.BatchNorm2d(f), nn.ReLU()]
            pool_w = 2 if width >= 2 else 1
            layers.append(nn.MaxPool2d((2, pool_w)) if i % 2 else nn.AvgPool2d((2, pool_w)))
            width //= pool_w
            channels = f
        layers.append(nn.Dropout(0.2))
        self.features = nn.Sequential(*layers)
        flat = channels * (144 // 2 ** len(filters)) * max(width, 1)
        head = [nn.Flatten()]
        head += [nn.Linear(flat, hidden), nn.ReLU(), nn.Linear(hidden, 1)] if hidden else [nn.Linear(flat, 1)]
        self.head = nn.Sequential(*head)

    def forward(self, x):  # x: (batch, 144, features)
        return self.head(self.features(x.unsqueeze(1))).squeeze(-1)


class BiLSTM(nn.Module):
    def __init__(self, n_features: int, units: int):
        super().__init__()
        self.lstm = nn.LSTM(n_features, units, batch_first=True, bidirectional=True)
        self.out = nn.Linear(2 * units, 1)

    def forward(self, x):  # x: (batch, 144, features)
        h, _ = self.lstm(x)
        return self.out(h[:, -1]).squeeze(-1)


class GRUBranch(nn.Module):
    """Two stacked GRUs (50 then 25 units); returns the last hidden state."""

    def __init__(self, n_features: int):
        super().__init__()
        self.gru1 = nn.GRU(n_features, 50, batch_first=True)
        self.gru2 = nn.GRU(50, 25, batch_first=True)

    def forward(self, x):
        h, _ = self.gru1(x)
        h, _ = self.gru2(h)
        return h[:, -1]


class GRUNet(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.gru = GRUBranch(n_features)
        self.out = nn.Linear(25, 1)

    def forward(self, x):
        return self.out(self.gru(x)).squeeze(-1)


class CNNGRU(nn.Module):
    def __init__(self, n_features: int, steps: int = 144):
        super().__init__()
        self.cnn = nn.Sequential(  # Conv1d works over time; the features are its channels
            nn.Conv1d(n_features, 32, kernel_size=2), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=2), nn.ReLU(),
            nn.MaxPool1d(2), nn.Flatten(),
        )
        self.gru = GRUBranch(n_features)
        self.out = nn.Linear(64 * ((steps - 2) // 2) + 25, 1)

    def forward(self, x):  # x: (batch, 144, features)
        both = torch.cat([self.cnn(x.transpose(1, 2)), self.gru(x)], dim=1)
        return self.out(both).squeeze(-1)


def make(name: str, n_features: int) -> nn.Module:
    if name == "CNN1":
        return CNN(n_features, (8, 16, 32, 64), hidden=512)
    if name == "CNN2":
        return CNN(n_features, (8, 16, 32, 32), hidden=None)
    if name == "RNN1":
        return BiLSTM(n_features, 64)
    if name == "RNN2":
        return BiLSTM(n_features, 128)
    if name == "GRU":
        return GRUNet(n_features)
    if name == "CNN-GRU":
        return CNNGRU(n_features)
    raise ValueError(name)


def fit_predict_deep(name: str, A: np.ndarray, target: np.ndarray, B: np.ndarray, seed: int = 0,
                     loss: str = "squared") -> np.ndarray:
    """A, B: scaled windows (n, 144, features); target: z-scored. Returns z-scored predictions for B.
    loss: "squared" (the papers) or "absolute" (predicts the typical hour; scripts/eval_forecast_habits.py)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    cut = int(len(A) * 0.9)  # last 10% of the training hours, in time order, for early stopping
    as_tensor = lambda a: torch.from_numpy(np.ascontiguousarray(a)).float().to(DEVICE)  # noqa: E731
    xt, yt = as_tensor(A[:cut]), as_tensor(target[:cut])
    xv, yv = as_tensor(A[cut:]), as_tensor(target[cut:])
    model = make(name, A.shape[2]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    loss_fn = nn.L1Loss() if loss == "absolute" else nn.MSELoss()
    best, best_state, waited = np.inf, None, 0
    for _ in range(MAX_EPOCHS):
        model.train()
        order = torch.from_numpy(rng.permutation(len(xt))).to(DEVICE)
        for i in range(0, len(order), BATCH):
            batch = order[i:i + BATCH]
            if len(batch) < 2:
                continue  # batch norm needs at least two samples
            opt.zero_grad()
            loss_fn(model(xt[batch]), yt[batch]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val = loss_fn(model(xv), yv).item() if len(xv) else 0.0
        plateau.step(val)
        if val < best - 1e-5:
            best, waited = val, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(as_tensor(B[i:i + 2048])).cpu().numpy()
                               for i in range(0, len(B), 2048)]) if len(B) else np.empty(0)
