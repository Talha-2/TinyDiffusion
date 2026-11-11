# TinyDiffusion

A denoising diffusion probabilistic model (DDPM) implemented from scratch in PyTorch,
following [Ho et al., 2020](https://arxiv.org/abs/2006.11239). No diffusion libraries:
the U-Net, noise schedules, training objective, and samplers are all written by hand.

## What's implemented

- **U-Net** with residual blocks, GroupNorm, multi-head self-attention at 16x16, and sinusoidal timestep embeddings
- **Noise schedules:** linear (Ho et al.) and cosine ([Nichol & Dhariwal, 2021](https://arxiv.org/abs/2102.09672))
- **Training:** simplified noise-prediction loss, EMA weights, mixed precision, gradient clipping, LR warmup, resumable checkpoints
- **Samplers:** full 1000-step DDPM ancestral sampling and [DDIM](https://arxiv.org/abs/2010.02502) for fast few-step sampling
- **Evaluation:** FID against the CIFAR-10 test set plus per-image sampling latency

## Results (CIFAR-10, 32x32)

Model: [X]M parameters, trained [X]k steps on a single [GPU].

| Schedule | Sampler | Steps | FID (10k samples) | ms / image |
|----------|---------|-------|-------------------|------------|
| Linear   | DDPM    | 1000  | [ ]               | [ ]        |
| Linear   | DDIM    | 250   | [ ]               | [ ]        |
| Linear   | DDIM    | 50    | [ ]               | [ ]        |
| Cosine   | DDIM    | 50    | [ ]               | [ ]        |

![samples](assets/samples.png)

## Findings

- [What you observed comparing linear vs cosine schedules]
- [How quality degrades as DDIM steps drop]
- [Failure cases you saw early in training]

## Reproduce

```bash
pip install -r requirements.txt
python train.py --out runs/linear --schedule linear --steps 100000
python train.py --out runs/cosine --schedule cosine --steps 100000
python evaluate.py --ckpt runs/linear/ckpt.pt --method ddim --steps 50
python sample.py --ckpt runs/linear/ckpt.pt --out assets/samples.png
```

Training is resumable: `python train.py --out runs/linear --resume runs/linear/ckpt.pt`
