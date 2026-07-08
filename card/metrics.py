from dataclasses import dataclass
from math import log10

import numpy as np

from card.types import FloatImage


@dataclass(frozen=True, slots=True)
class ImageMetrics:
    psnr: float
    ssim: float


def compute_metrics(reference: FloatImage, candidate: FloatImage) -> ImageMetrics:
    mse = float(np.mean((reference - candidate) ** 2))
    psnr = 99.0 if mse <= 0.0 else 10.0 * log10(1.0 / mse)
    ssim = _global_ssim(reference, candidate)
    return ImageMetrics(psnr=psnr, ssim=ssim)


def _global_ssim(reference: FloatImage, candidate: FloatImage) -> float:
    c1 = 0.01**2
    c2 = 0.03**2
    ref_mean = float(np.mean(reference))
    cand_mean = float(np.mean(candidate))
    ref_var = float(np.var(reference))
    cand_var = float(np.var(candidate))
    covariance = float(np.mean((reference - ref_mean) * (candidate - cand_mean)))
    numerator = (2 * ref_mean * cand_mean + c1) * (2 * covariance + c2)
    denominator = (ref_mean**2 + cand_mean**2 + c1) * (ref_var + cand_var + c2)
    return numerator / denominator
