import argparse
import torch
from torchvision.utils import save_image

from diffusion import GaussianDiffusion
from model import build_model


def main():
    p = argparse.ArgumentParser(description="Generate samples from a trained checkpoint")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--method", choices=["ddpm", "ddim"], default="ddim")
    p.add_argument("--steps", type=int, default=50, help="DDIM steps (ignored for ddpm)")
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="samples.png")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=device)
    cfg = ck["cfg"]
    model = build_model(cfg).to(device)
    model.load_state_dict(ck["ema"])
    model.eval()
    diffusion = GaussianDiffusion(cfg["timesteps"], cfg["schedule"]).to(device)

    x = diffusion.sample(model, (args.n, 3, 32, 32), method=args.method, steps=args.steps)
    save_image((x + 1) / 2, args.out, nrow=8)
    print(f"Saved {args.n} samples to {args.out}")


if __name__ == "__main__":
    main()
