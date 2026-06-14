import torch
import torch.nn as nn


class CLUBLoss(nn.Module):
    """Contrastive Log-ratio Upper Bound on I(X; Y) — Cheng et al., ICML 2020."""

    def __init__(self, x_dim, y_dim, hidden_dim=None):
        super().__init__()
        h = hidden_dim or max(x_dim, y_dim)
        self.mu_net = nn.Sequential(
            nn.Linear(x_dim, h), nn.ReLU(),
            nn.Linear(h, y_dim),
        )
        self.logvar_net = nn.Sequential(
            nn.Linear(x_dim, h), nn.ReLU(),
            nn.Linear(h, y_dim), nn.Tanh(),
        )

    def _q_params(self, x):
        return self.mu_net(x), self.logvar_net(x)

    @staticmethod
    def _log_gauss(y, mu, logvar):
        return -0.5 * (logvar + (y - mu).pow(2) * (-logvar).exp())

    def forward(self, x, y):
        mu, logvar = self._q_params(x)
        positive = self._log_gauss(y, mu, logvar).sum(-1)
        diff = y.unsqueeze(0) - mu.unsqueeze(1)
        lv = logvar.unsqueeze(1)
        negative = (-0.5 * (lv + diff.pow(2) * (-lv).exp())).sum(-1).mean(-1)
        return (positive - negative).mean()

    def learning_loss(self, x, y):
        mu, logvar = self._q_params(x)
        return -self._log_gauss(y, mu, logvar).sum(-1).mean()
