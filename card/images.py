from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
from PIL import Image

from card.metrics import ImageMetrics
from card.types import FloatImage

DISK_RADIUS_SQUARED: Final = 0.09
SQUARE_X_MIN: Final = 0.12
SQUARE_X_MAX: Final = 0.38
SQUARE_Y_MIN: Final = 0.58
SQUARE_Y_MAX: Final = 0.82


@dataclass(frozen=True, slots=True)
class ImageGrid:
    clean: FloatImage
    noisy: FloatImage
    baseline: FloatImage
    card: FloatImage
    metrics: dict[str, ImageMetrics]


def load_sample_image(size: int) -> FloatImage:
    coords = np.linspace(0.0, 1.0, size, dtype=np.float64)
    x_grid, y_grid = np.meshgrid(coords, coords)
    gradient = 0.35 * x_grid + 0.25 * y_grid
    disk = ((x_grid - 0.68) ** 2 + (y_grid - 0.35) ** 2) < DISK_RADIUS_SQUARED
    square = (
        (x_grid > SQUARE_X_MIN)
        & (x_grid < SQUARE_X_MAX)
        & (y_grid > SQUARE_Y_MIN)
        & (y_grid < SQUARE_Y_MAX)
    )
    stripes = 0.12 * (np.sin(34 * x_grid) > 0)
    image = gradient + stripes
    image[disk] += 0.35
    image[square] -= 0.25
    return np.asarray(np.clip(image, 0.0, 1.0), dtype=np.float64)


def save_image_grid(output_path: Path, grid: ImageGrid) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panels = [
        ("Clean", grid.clean),
        ("Correlated noisy", grid.noisy),
        ("Gaussian baseline", grid.baseline),
        ("CARD-style restore", grid.card),
    ]
    tile_height = len(grid.clean)
    tile_width = len(grid.clean[0])
    label_height = 24
    canvas = np.ones(
        (tile_height + label_height, tile_width * len(panels)),
        dtype=np.float64,
    )
    for index, (title, image) in enumerate(panels):
        start = index * tile_width
        canvas[label_height:, start : start + tile_width] = image
        _draw_metric_bar(canvas[:label_height, start : start + tile_width], title, grid.metrics)
    pixels = np.asarray(np.clip(canvas, 0.0, 1.0) * 255, dtype=np.uint8)
    Image.fromarray(pixels).save(output_path)


def _draw_metric_bar(
    canvas: FloatImage,
    title: str,
    metrics: dict[str, ImageMetrics],
) -> None:
    metric = metrics.get(title)
    intensity = 0.72
    if metric is not None:
        intensity = min(0.95, max(0.25, metric.psnr / 45.0))
    canvas[:, :] = intensity
