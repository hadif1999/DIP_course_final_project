# CARD Course Implementation

This repository contains a course implementation and presentation package for
`CARD: Correlation Aware Restoration with Diffusion` (`card_paper.pdf`).

CARD addresses image restoration when camera noise is spatially correlated
rather than independent. The paper's measurement model is:

```text
y = Hx0 + n
n ~ N(0, sigma_y^2 Sigma)
```

The key idea is to whiten the measurement equation:

```text
W = Sigma^(-1/2)
y_tilde = W y
H_tilde = W H
W Sigma W^T = I
```

Then DDRM-style diffusion restoration is applied in the spectral basis of
`H_tilde`, not the original `H`.

## Repository Contents

- `card_paper.pdf` - source paper.
- `card/` - small CPU implementation of covariance modeling, whitening, and a
  covariance-aware denoising demo.
- `main.py` - runs the CPU demo and writes `results/card_demo.png`.
- `card_guided_diffusion_colab.ipynb` - Google Colab diffusion implementation
  closer to the paper methodology.
- `report.tex` / `report.pdf` - paper-aligned written report.
- `presentation.tex` / `presentation.pdf` - paper-aligned slide deck.
- `tests/` - focused tests for whitening and the CPU pipeline.

## Implementation Scope

This project has two implementation levels.

### CPU demo

The Python package is intentionally compact and reproducible on CPU. It
demonstrates:

- patchwise correlated Gaussian noise;
- covariance matrix construction;
- coloring transform for generating correlated noise;
- whitening transform with `W Sigma W^T ~= I`;
- Gaussian denoising baseline;
- covariance-aware restoration baseline;
- PSNR and global SSIM metrics.

Run it with:

```bash
uv sync
uv run python main.py
```

Output:

```text
results/card_demo.png
```

### Colab diffusion implementation

For the version closest to the paper implementation, use:

```text
card_guided_diffusion_colab.ipynb
```

Run it in Google Colab with GPU enabled. The notebook:

- clones OpenAI `guided-diffusion`;
- downloads the public `64x64_diffusion.pt` checkpoint;
- builds a patchwise covariance matrix;
- generates correlated Gaussian measurements;
- computes the whitening transform;
- uses the SVD of `H_tilde = W H`;
- restores multiple denoising test images with a frozen diffusion prior;
- compares noisy, Gaussian baseline, and CARD-style restored outputs.

The notebook focuses on denoising, so `H = I` and `H_tilde = W`.

## Deliverables

The report and presentation are already compiled:

```text
report.pdf        # 7 pages
presentation.pdf  # 15 slides
```

Rebuild them with:

```bash
pdflatex -interaction=nonstopmode -halt-on-error report.tex
pdflatex -interaction=nonstopmode -halt-on-error report.tex
pdflatex -interaction=nonstopmode -halt-on-error presentation.tex
pdflatex -interaction=nonstopmode -halt-on-error presentation.tex
```

## Verification

```bash
uv run pytest
uv run ruff check .
uv run basedpyright
```

Current expected status:

```text
4 passed
All checks passed
0 errors, 0 warnings, 0 notes
```

## Relation to the Paper

Aligned with `card_paper.pdf`:

- correlated Gaussian noise model;
- covariance-aware restoration problem;
- patchwise covariance and whitening;
- synthetic correlated-noise generation;
- frozen diffusion prior in the Colab notebook;
- whitened spectral measurement update for the denoising case.

Not yet implemented:

- full original DDRM codebase integration;
- real dark-frame covariance estimation;
- CIN-D evaluation;
- LPIPS metric;
- deblurring and super-resolution tasks.

So the repo should be presented as a faithful course-scale implementation of the
CARD methodology for denoising, plus a compact CPU demonstration for inspection
and reproducibility.
