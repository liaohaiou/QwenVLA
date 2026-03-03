import torch
import math

class Scheduler:
    def __init__(self, cfg : dict):
        self.T = cfg['timesteps']
        self.offset = cfg['offset']

        # compute in float32 for stability
        self.beta = torch.zeros(self.T)
        self.alpha = torch.zeros(self.T)      # per-step alpha
        self.alphabar = torch.zeros(self.T)   # cumulative

        self._build_schedule()

    def _build_schedule(self):
        t = torch.arange(self.T)

        # cosine cumulative schedule (your alpha0)
        alphabar = torch.cos(((t / self.T + self.offset) / (1 + self.offset)) * math.pi / 2) ** 2
        alphabar = alphabar / alphabar[0]  # normalize so alphabar[0]=1

        self.alphabar = alphabar.clamp(1e-6, 1.0)

        # per-step alpha and beta
        self.alpha[0] = self.alphabar[0]
        self.alpha[1:] = self.alphabar[1:] / self.alphabar[:-1]
        self.beta = (1.0 - self.alpha).clamp(1e-6, 0.999)

    def add_noise(self, x0, t, eps=None):
        """
        x0: (..., traj_dims)
        t:  int or tensor of indices
        eps: optional noise same shape as x0
        """
        if eps is None:
            eps = torch.randn_like(x0)

        ab = self.alphabar[t].view(*([1] * (x0.ndim - 1)), 1).to(device=x0.device, dtype=x0.dtype) # broadcast
        xt = torch.sqrt(ab) * x0.float() + torch.sqrt(1.0 - ab) * eps
        return xt, eps

    def step(self, xt, eps_pred, t, add_variance=False):
        """
        xt: (..., traj_dims)
        eps_pred: predicted noise same shape
        """
        a = self.alpha[t]
        b = self.beta[t]
        ab = self.alphabar[t]

        a = a.view(*([1] * (xt.ndim - 1)), 1)
        b = b.view(*([1] * (xt.ndim - 1)), 1)
        ab = ab.view(*([1] * (xt.ndim - 1)), 1)

        mean = (1.0 / torch.sqrt(a)) * (xt - (b / torch.sqrt(1.0 - ab)) * eps_pred)

        if not add_variance or t == 0:
            return mean

        # DDPM variance (one common choice)
        z = torch.randn_like(xt)
        sigma = torch.sqrt(b)  # simplified; some implementations use posterior variance
        return mean + sigma * z