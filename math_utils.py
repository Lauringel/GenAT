import numpy as np
import torch.nn.functional as F

# Spatial gradient
def spatial_grad(x):
    """
    x: (B, C, H, W)
    """
    grad_x = x[:, :, :, 1:] - x[:, :, :, :-1]
    grad_y = x[:, :, 1:, :] - x[:, :, :-1, :]

    # pad to original size
    grad_x = F.pad(grad_x, (0, 1, 0, 0))
    grad_y = F.pad(grad_y, (0, 0, 0, 1))
    return grad_x, grad_y

# Squared L2-norm of gradient \|grad\|^2
def gradient_l2(x):
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    return dx.pow(2).mean() + dy.pow(2).mean()

# Squared weighted L2-norm of gradient ||v grad||^2
def gradient_G_weighted(Gz, v):
    dx = Gz[:, :, :, 1:] - Gz[:, :, :, :-1]
    dy = Gz[:, :, 1:, :] - Gz[:, :, :-1, :]
    vx = v[:, :, :, 1:].expand_as(dx)
    vy = v[:, :, 1:, :].expand_as(dy)
    return (vx.pow(2) * dx.pow(2)).mean() + (vy.pow(2) * dy.pow(2)).mean()

# Spatial gradient for 3 channels
def grad_rgb(U):
    """
    Compute forward differences per channel
    U: shape (3,H,W)
    Returns dx, dy: shape (3,H,W)
    """
    dx = np.roll(U, -1, axis=2) - U
    dy = np.roll(U, -1, axis=1) - U
    return dx, dy

# Divergence for RGB
def div_rgb(px, py):
    """
    Compute divergence per channel
    px, py: shape (3,H,W)
    """
    mx = px - np.roll(px, 1, axis=2)
    my = py - np.roll(py, 1, axis=1)
    return mx + my
