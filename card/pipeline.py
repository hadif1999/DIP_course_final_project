from dataclasses import dataclass

import numpy as np

from card.images import ImageGrid, load_sample_image, save_image_grid
from card.metrics import ImageMetrics, compute_metrics
from card.noise import (
    add_correlated_noise,
    build_noise_model,
    make_rolling_shutter_covariance,
)
from card.restoration import card_wiener_restore, gaussian_baseline
from card.types import DemoConfig


@dataclass(frozen=True, slots=True)
class DemoResult:
    metrics: dict[str, ImageMetrics]
    figure_path: str


def run_demo(config: DemoConfig) -> DemoResult:
    rng = np.random.default_rng(config.seed)
    covariance = make_rolling_shutter_covariance(
        config.patch_size,
        config.correlation_strength,
        config.regularization,
    )
    model = build_noise_model(covariance)
    clean = load_sample_image(config.image_size)
    noisy = add_correlated_noise(clean, model, config.noise_sigma, rng)
    baseline = gaussian_baseline(noisy)
    card = card_wiener_restore(noisy, model, config.noise_sigma)
    metrics = {
        "Correlated noisy": compute_metrics(clean, noisy),
        "Gaussian baseline": compute_metrics(clean, baseline),
        "CARD-style restore": compute_metrics(clean, card),
    }
    figure_path = config.output_dir / "card_demo.png"
    save_image_grid(
        figure_path,
        ImageGrid(clean=clean, noisy=noisy, baseline=baseline, card=card, metrics=metrics),
    )
    return DemoResult(metrics=metrics, figure_path=str(figure_path))
