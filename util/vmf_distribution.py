import torch
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.learning.frechet_mean import FrechetMean


class SteinVMFDistribution:
    def __init__(self, dtype: torch.dtype = torch.float64):
        self.dtype = dtype
        self.kappa: torch.Tensor = None
        self.mu: torch.Tensor = None  # (d, 1)

    def fit(self, x_sample: torch.Tensor, eps: float = 1e-6):
        # x_sample: (N, d)
        N, d = x_sample.shape
        device = x_sample.device

        x_sample = x_sample.reshape(N, d, 1).to(dtype=self.dtype)  # (N, d, 1)

        mu = torch.tensor(
            (
                FrechetMean(Hypersphere(dim=d - 1))
                .fit(x_sample.squeeze(-1).cpu().numpy())
                .estimate_
            ),
            dtype=self.dtype,
            device=device,
        ).unsqueeze(-1)  # (d, 1)
        if torch.any(torch.isnan(mu)) or torch.any(torch.isinf(mu)):
            mu = x_sample.mean(dim=0)  # (d, 1)
            mu /= torch.norm(mu, p=2, dim=-2, keepdim=True).clip(min=eps)

        x_mean = x_sample.mean(dim=0)  # (d, 1)
        P_mean = torch.eye(d, dtype=self.dtype, device=device) - (
            x_sample @ x_sample.transpose(-1, -2)
        ).mean(dim=0)  # (d, d)

        kappa = ((d - 1) * mu.transpose(-1, -2) @ P_mean @ x_mean) / (
            mu.transpose(-1, -2) @ P_mean @ P_mean @ mu
        ).clip(min=eps).squeeze()

        self.kappa = kappa
        self.mu = mu  # (d, 1)

        return self

    def log_pdf(self, x_query: torch.Tensor):
        # x_query: (N, d)

        x_query = x_query.to(dtype=self.dtype)

        log_pdf = (
            self.kappa * self.mu.reshape(1, 1, -1) @ x_query.unsqueeze(-1)
        ).squeeze()  # (N,)

        return log_pdf

    def mode(self):
        x = self.mu.squeeze(-1)  # (d,)
        return x
