import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import laplace

from math_utils import grad_rgb, spatial_grad


def ambrosio_tortorelli_rgb(F, mu, eps, lam_data, n_iter, dt):
    H, W, C = F.shape
    U = F.copy().transpose(2, 0, 1)
    v = np.ones((H, W))

    for _ in range(n_iter):
        dx, dy = grad_rgb(U)
        grad_sq = np.sum(dx ** 2 + dy ** 2, axis=0)
        grad_sq = np.minimum(grad_sq, 1e4)

        v_lap = laplace(v)

        v = (
            v + dt * (
                -2 * lam_data * grad_sq * v
                + mu * eps * v_lap
                + mu / (2 * eps)
            )
        ) / (1 + dt * (mu / (2 * eps)))

        v = np.clip(v, 0, 1)

        for c in range(C):
            ux, uy = dx[c], dy[c]

            div_term = (
                np.roll(v * v * ux, 1, axis=1) - v * v * ux
                + np.roll(v * v * uy, 1, axis=0) - v * v * uy
            )

            U[c] = (
                U[c] + dt * (F[:, :, c] + lam_data * div_term)
            ) / (1 + dt)

        U = np.clip(U, 0.0, 1.0)

    return U.transpose(1, 2, 0), v


class SimpleATDetector(nn.Module):
    def __init__(self, mu, eps, lam_AT, dt):
        super().__init__()
        self.mu = mu
        self.eps = eps
        self.lam_AT = lam_AT
        self.dt = dt

    def forward(self, y, iters):
        B, C, H, W = y.shape

        if B != 1:
            raise ValueError("SimpleATDetector supports only batch size 1.")

        device = y.device
        img_np = y[0].detach().cpu().permute(1, 2, 0).numpy()

        U_np, v_np = ambrosio_tortorelli_rgb(
            img_np,
            mu=self.mu,
            eps=self.eps,
            lam_data=self.lam_AT,
            n_iter=iters,
            dt=self.dt
        )

        v = torch.from_numpy(v_np).unsqueeze(0).unsqueeze(0).to(device=y.device, dtype=y.dtype)
        U = torch.from_numpy(U_np).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=y.dtype)

        return U.detach(), v.detach()


class SimpleMaskedATDetector(nn.Module):
    def __init__(self, mu, eps, lam_AT):
        super().__init__()
        self.mu = mu
        self.eps = eps
        self.lam_AT = lam_AT

    def forward(self, y, iters, lr_u=0.01, lr_v=0.1):
        device = y.device
        B, C, H, W = y.shape
        assert B == 1

        U = y.clone().detach().to(device).requires_grad_(True)

        s = torch.zeros(
            (B, 1, H, W),
            device=device,
            requires_grad=True
        )

        opt = torch.optim.Adam([
            {"params": [U], "lr": lr_u},
            {"params": [s], "lr": lr_v}
        ])

        for _ in range(iters):
            opt.zero_grad()

            v = torch.sigmoid(s)
            v_rgb = v.expand(-1, C, -1, -1)

            data_term = ((v_rgb * (y - U)) ** 2).mean()

            ux, uy = spatial_grad(U)
            grad_sq = (ux ** 2 + uy ** 2).sum(dim=1, keepdim=True)
            energy_bulk = (v * v * grad_sq).mean()

            vx, vy = spatial_grad(v)

            energy_v = (
                self.eps * (vx ** 2 + vy ** 2).mean()
                + ((v - 1) ** 2 / (4 * self.eps)).mean()
            )

            energy_AT = energy_bulk + self.mu * energy_v

            loss = data_term + self.lam_AT * energy_AT

            loss.backward()
            opt.step()

            with torch.no_grad():
                U.clamp_(0.0, 1.0)

        return U.detach(), torch.sigmoid(s).detach()
