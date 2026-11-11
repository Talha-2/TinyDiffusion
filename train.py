import argparse
import copy
import os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm

from diffusion import GaussianDiffusion
from model import build_model, count_params


def parse_args():
    p = argparse.ArgumentParser(description="Train a DDPM on CIFAR-10")
    p.add_argument("--out", default="runs/linear")
    p.add_argument("--data", default="data")
    p.add_argument("--schedule", choices=["linear", "cosine"], default="linear")
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--ch", type=int, default=64)
    p.add_argument("--ch-mult", default="1,2,2,2")
    p.add_argument("--num-res", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup", type=int, default=5000)
    p.add_argument("--ema-decay", type=float, default=0.9999)
    p.add_argument("--sample-every", type=int, default=5000)
    p.add_argument("--save-every", type=int, default=5000)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--resume", default=None, help="path to ckpt.pt to continue training")
    return p.parse_args()


def infinite(loader):
    while True:
        for x, _ in loader:
            yield x


@torch.no_grad()
def ema_update(ema, model, decay):
    for e, p in zip(ema.parameters(), model.parameters()):
        e.lerp_(p, 1 - decay)


def main():
    args = parse_args()
    cfg = vars(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)
    torch.backends.cudnn.benchmark = True

    tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3),  # scale to [-1, 1]
    ])
    ds = datasets.CIFAR10(args.data, train=True, download=True, transform=tf)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=4, drop_last=True, pin_memory=True)

    model = build_model(cfg).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    diffusion = GaussianDiffusion(args.timesteps, args.schedule).to(device)
    print(f"U-Net parameters: {count_params(model):.1f}M")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup))
    use_amp = device == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    step = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        step = ck["step"]
        print(f"Resumed from step {step}")

    if args.wandb:
        import wandb
        wandb.init(project="tinydiffusion", name=os.path.basename(args.out), config=cfg)

    data = infinite(loader)
    pbar = tqdm(total=args.steps, initial=step)
    model.train()
    while step < args.steps:
        x0 = next(data).to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            loss = diffusion.loss(model, x0)

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        ema_update(ema, model, args.ema_decay)
        step += 1
        pbar.update(1)

        if step % 100 == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            if args.wandb:
                wandb.log({"loss": loss.item(), "lr": sched.get_last_lr()[0]}, step=step)

        if step % args.sample_every == 0:
            x = diffusion.ddim_sample(ema, (64, 3, 32, 32), steps=50)
            path = os.path.join(args.out, f"samples_{step:07d}.png")
            save_image((x + 1) / 2, path, nrow=8)
            if args.wandb:
                wandb.log({"samples": wandb.Image(path)}, step=step)

        if step % args.save_every == 0 or step == args.steps:
            torch.save({
                "model": model.state_dict(), "ema": ema.state_dict(),
                "opt": opt.state_dict(), "sched": sched.state_dict(),
                "scaler": scaler.state_dict(), "step": step, "cfg": cfg,
            }, os.path.join(args.out, "ckpt.pt"))


if __name__ == "__main__":
    main()
