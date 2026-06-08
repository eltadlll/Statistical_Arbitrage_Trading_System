"""
models/lstm_selector.py
-----------------------
LSTM-based model that learns regime-conditioned spread behaviour.
Accepts a rolling feature sequence and predicts whether the pair
is in a favourable regime for statistical arbitrage over the next window.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

# Lazy import so the module loads even without torch installed
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not found – LSTMSelector will raise if used.")


class _LSTMModel(nn.Module if TORCH_AVAILABLE else object):
    """Stacked LSTM with dropout and a binary classification head."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])   # last timestep
        return self.sigmoid(self.fc(out)).squeeze(1)


class LSTMPairSelector:
    """
    Trains an LSTM to detect favourable regimes for each pair.

    Parameters
    ----------
    lookback    : Sequence length (days) fed into the LSTM.
    hidden_size : LSTM hidden units.
    num_layers  : Stacked LSTM layers.
    epochs      : Training epochs.
    batch_size  : Mini-batch size.
    lr          : Adam learning rate.
    device      : "cpu" | "cuda" | "auto"
    """

    def __init__(
        self,
        lookback: int = 60,
        hidden_size: int = 64,
        num_layers: int = 2,
        epochs: int = 30,
        batch_size: int = 32,
        lr: float = 1e-3,
        device: str = "auto",
    ) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError("Install torch to use LSTMPairSelector.")

        self.lookback = lookback
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model: Optional[_LSTMModel] = None
        self.feature_cols: list[str] = []

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def prepare_sequences(
        self,
        feature_df: pd.DataFrame,
        label_col: str = "label",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert a flat feature DataFrame into overlapping sequences.

        Returns (X, y) arrays with shapes (n_samples, lookback, n_features)
        and (n_samples,) respectively.
        """
        self.feature_cols = [c for c in feature_df.columns if c != label_col]
        data = feature_df[self.feature_cols].values.astype(np.float32)
        labels = feature_df[label_col].values.astype(np.float32)

        X, y = [], []
        for i in range(self.lookback, len(data)):
            X.append(data[i - self.lookback:i])
            y.append(labels[i])

        return np.array(X), np.array(y)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "LSTMPairSelector":
        """Train the LSTM on pre-built sequences."""
        input_size = X_train.shape[2]
        self.model = _LSTMModel(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
        ).to(self.device)

        dataset = TensorDataset(
            torch.tensor(X_train), torch.tensor(y_train)
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimiser = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.BCELoss()

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimiser.zero_grad()
                preds = self.model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimiser.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(loader)

            if X_val is not None and epoch % 5 == 0:
                val_acc = self._evaluate(X_val, y_val)
                logger.info(
                    f"  Epoch {epoch}/{self.epochs} | loss {avg_loss:.4f} | val_acc {val_acc:.3f}"
                )
            elif epoch % 5 == 0:
                logger.info(f"  Epoch {epoch}/{self.epochs} | loss {avg_loss:.4f}")

        logger.info("LSTM training complete.")
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability of class 1 for each sequence in X."""
        if self.model is None:
            raise RuntimeError("Call fit() first.")
        self.model.eval()
        with torch.no_grad():
            tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            probs = self.model(tensor).cpu().numpy()
        return probs

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        if self.model is None:
            raise RuntimeError("No model to save.")
        torch.save(self.model.state_dict(), path)
        logger.info(f"LSTM weights saved → {path}")

    def load(self, path: Path, input_size: int) -> "LSTMPairSelector":
        self.model = _LSTMModel(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
        ).to(self.device)
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        return self

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """Binary accuracy on a validation set."""
        probs = self.predict_proba(X)
        preds = (probs >= 0.5).astype(int)
        return float((preds == y.astype(int)).mean())
