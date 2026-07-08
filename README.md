# CARD: Correlation Aware Restoration with Diffusion

This project studies and implements the main idea from `card_paper.pdf`. The
problem is image restoration when the noise is spatially correlated, which is a
common situation in real camera sensors, especially rolling-shutter sensors.

Many restoration methods assume independent Gaussian noise:

```text
y = Hx0 + z
z ~ N(0, sigma_y^2 I)
```

CARD uses a more realistic correlated-noise model:

```text
y = Hx0 + n
n ~ N(0, sigma_y^2 Sigma)
```

The covariance matrix `Sigma` describes the relation between neighboring noise
values. CARD whitens the measurement equation:

```text
W = Sigma^(-1/2)
y_tilde = W y
H_tilde = W H
W Sigma W^T = I
```

After whitening, the noise becomes independent again, so diffusion restoration
can be applied in the spectral domain of `H_tilde`.

## Files

- `card_paper.pdf` - reference paper.
- `card/` - Python implementation for covariance modeling, whitening, and a CPU
  denoising experiment.
- `main.py` - runs the CPU experiment.
- `card_guided_diffusion_colab.ipynb` - Google Colab notebook using a pretrained
  diffusion model.
- `report.tex` / `report.pdf` - written project report.
- `presentation.tex` / `presentation.pdf` - presentation slides.
- `tests/` - tests for whitening and the demo pipeline.

## CPU Experiment

The CPU version is a compact implementation of the image-processing part of the
method. It demonstrates:

- synthetic correlated Gaussian noise;
- patchwise covariance construction;
- coloring and whitening transforms;
- a Gaussian denoising baseline;
- a covariance-aware restoration baseline;
- PSNR and global SSIM measurements.

Run:

```bash
uv sync
uv run python main.py
```

The output image is saved to:

```text
results/card_demo.png
```

## Colab Diffusion Experiment

The Colab notebook uses a pretrained OpenAI guided-diffusion model and applies
the covariance-aware restoration idea to denoising. In this experiment the
degradation operator is identity:

```text
H = I
H_tilde = W
```

The notebook:

- builds a patchwise covariance matrix;
- generates correlated noisy measurements;
- computes the whitening matrix;
- uses the singular value decomposition of `H_tilde`;
- restores several test images;
- compares noisy, Gaussian baseline, and restored images.

Run `card_guided_diffusion_colab.ipynb` in Google Colab with GPU enabled.

## Report and Presentation

The report and slides are included as both source and compiled PDF files:

```text
report.tex
report.pdf
presentation.tex
presentation.pdf
```

To rebuild them:

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

Expected result:

```text
4 passed
All checks passed
0 errors, 0 warnings, 0 notes
```

## Current Scope

Implemented:

- correlated Gaussian noise model;
- patchwise covariance and whitening;
- synthetic correlated-noise generation;
- CPU denoising experiment;
- Colab diffusion denoising experiment;
- PSNR and global SSIM evaluation.

Not included in this version:

- full DDRM codebase reproduction;
- real dark-frame covariance estimation;
- CIN-D dataset evaluation;
- LPIPS metric;
- deblurring and super-resolution experiments.
