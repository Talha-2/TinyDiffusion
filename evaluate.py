import argparse
import time

import torch
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import datasets, transforms

from diffusion import GaussianDiffusion
from model import build_model


def main():
    p = argparse.ArgumentParser(description="FID and sampling speed vs CIFAR-10 test set")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--method", choices=["ddpm", "ddim"], default="ddim")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--n", type=int, default=10_000)
    p.add_argument("--batch", type=int, default=500)
    p.add_argument("--data", default="data")
    args = p.parse_args()

    device = "cuda"
    ck = torch.load(args.ckpt, map_location=device)
    cfg = ck["cfg"]
    model = build_model(cfg).to(device)
    model.load_state_dict(ck["ema"])
    model.eval()
    diffusion = GaussianDiffusion(cfg["timesteps"], cfg["schedule"]).to(device)

    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    real = datasets.CIFAR10(args.data, train=False, download=True, transform=transforms.ToTensor())
    for x, _ in DataLoader(real, batch_size=500):
        fid.update(x.to(device), real=True)

    done, sample_time = 0, 0.0
    while done < args.n:
        b = min(args.batch, args.n - done)
        torch.cuda.synchronize()
        t0 = time.time()
        x = diffusion.sample(model, (b, 3, 32, 32), method=args.method, steps=args.steps)
        torch.cuda.synchronize()
        sample_time += time.time() - t0
        fid.update(((x + 1) / 2).clamp(0, 1), real=False)
        done += b
        print(f"{done}/{args.n} generated")

    steps = cfg["timesteps"] if args.method == "ddpm" else args.steps
    print(f"schedule={cfg['schedule']} method={args.method} steps={steps} "
          f"FID={fid.compute().item():.2f} ms/image={sample_time / args.n * 1000:.1f}")


if __name__ == "__main__":
    main()
