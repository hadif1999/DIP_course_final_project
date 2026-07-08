# CARD Course Implementation

This project is a compact, reproducible implementation of the core idea from
`CARD: Correlation Aware Restoration with Diffusion`.

It does **not** implement the full diffusion/DDRM sampler. Instead, it isolates
the image-processing idea that makes CARD useful: model spatially correlated
sensor noise with a covariance matrix, use that covariance to whiten or
diagonalize the noise, and restore in that correlation-aware space.

## What It Demonstrates

- Synthetic rolling-shutter-like correlated Gaussian noise.
- Patch-local covariance matrix construction.
- Covariance whitening.
- A correlation-aware Wiener restoration baseline.
- A standard Gaussian denoising baseline that ignores correlation.
- PSNR/SSIM metrics and a saved comparison figure.

## Run

```bash
uv sync
uv run python main.py
```

The demo writes:

```text
results/card_demo.png
```

## Verify

```bash
uv run pytest
uv run ruff check .
uv run basedpyright
```

## Relation To The Paper

The paper's full CARD method uses a pretrained diffusion prior and modifies DDRM
updates after whitening the measurement equation. This course implementation
keeps the same covariance-aware restoration motivation, but replaces the
diffusion prior with a lightweight Wiener shrinkage step so it can run quickly on
CPU and be inspected easily.
