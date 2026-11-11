import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_beta_schedule(kind, T):
    if kind == "linear":  # Ho et al., 2020
        return torch.linspace(1e-4, 0.02, T)
    if kind == "cosine":  # Nichol & Dhariwal, 2021
        s = 0.008
        steps = torch.arange(T + 1, dtype=torch.float64) / T
        alpha_bar = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
        return betas.clamp(max=0.999).float()
    raise ValueError(f"unknown schedule: {kind}")


def _extract(a, t, x):
    return a[t].view(-1, *([1] * (x.dim() - 1)))


class GaussianDiffusion(nn.Module):
    def __init__(self, T=1000, schedule="linear"):
        super().__init__()
        self.T = T
        betas = make_beta_schedule(schedule, T)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_alpha_bar", alpha_bar.sqrt())
        self.register_buffer("sqrt_one_minus_alpha_bar", (1 - alpha_bar).sqrt())

    def q_sample(self, x0, t, noise):
        """Forward process: jump straight to x_t from x_0 in closed form."""
        return _extract(self.sqrt_alpha_bar, t, x0) * x0 + _extract(self.sqrt_one_minus_alpha_bar, t, x0) * noise

    def loss(self, model, x0):
        """Simplified training objective: predict the added noise."""
        t = torch.randint(0, self.T, (x0.shape[0],), device=x0.device)
        noise = torch.randn_like(x0)
        pred = model(self.q_sample(x0, t, noise), t)
        return F.mse_loss(pred.float(), noise)

    @torch.no_grad()
    def ddpm_sample(self, model, shape):
        """Ancestral sampling over all T steps (Algorithm 2 in the DDPM paper)."""
        device = next(model.parameters()).device
        x = torch.randn(shape, device=device)
        for i in reversed(range(self.T)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            eps = model(x, t)
            beta, alpha, ab = self.betas[i], self.alphas[i], self.alpha_bar[i]
            mean = (x - beta / (1 - ab).sqrt() * eps) / alpha.sqrt()
            x = mean + beta.sqrt() * torch.randn_like(x) if i > 0 else mean
        return x.clamp(-1, 1)

    @torch.no_grad()
    def ddim_sample(self, model, shape, steps=50, eta=0.0):
        """DDIM sampling (Song et al., 2021): far fewer steps, deterministic when eta=0."""
        device = next(model.parameters()).device
        ts = torch.linspace(self.T - 1, 0, steps).long().tolist()
        x = torch.randn(shape, device=device)
        for i, t in enumerate(ts):
            ab_t = self.alpha_bar[t]
            ab_prev = self.alpha_bar[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=device)
            eps = model(x, torch.full((shape[0],), t, device=device, dtype=torch.long))
            x0 = ((x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()).clamp(-1, 1)
            sigma = eta * ((1 - ab_prev) / (1 - ab_t) * (1 - ab_t / ab_prev)).sqrt()
            x = ab_prev.sqrt() * x0 + (1 - ab_prev - sigma ** 2).sqrt() * eps + sigma * torch.randn_like(x)
        return x.clamp(-1, 1)

    def sample(self, model, shape, method="ddim", steps=50):
        if method == "ddpm":
            return self.ddpm_sample(model, shape)
        return self.ddim_sample(model, shape, steps=steps)
