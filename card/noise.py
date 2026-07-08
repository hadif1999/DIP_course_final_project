from dataclasses import dataclass
from math import isqrt

import numpy as np

from card.types import FloatImage


@dataclass(frozen=True, slots=True)
class NoiseModel:
    patch_size: int
    covariance: FloatImage
    whitening: FloatImage
    coloring: FloatImage


def make_rolling_shutter_covariance(
    patch_size: int,
    correlation_strength: float,
    regularization: float,
) -> FloatImage:
    coords = np.indices((patch_size, patch_size), dtype=np.float64).reshape(2, -1).T
    row_distance = np.abs(coords[:, 0, None] - coords[None, :, 0])
    col_distance = np.abs(coords[:, 1, None] - coords[None, :, 1])
    row_term = correlation_strength**row_distance
    col_term = (correlation_strength * 0.45) ** col_distance
    covariance = row_term * col_term
    return covariance + regularization * np.eye(patch_size * patch_size)


def build_noise_model(covariance: FloatImage) -> NoiseModel:
    patch_size = isqrt(len(covariance))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    clipped = np.maximum(eigenvalues, 1e-10)
    whitening = np.asarray(
        eigenvectors @ np.diag(1.0 / np.sqrt(clipped)) @ eigenvectors.T,
        dtype=np.float64,
    )
    coloring = np.asarray(
        eigenvectors @ np.diag(np.sqrt(clipped)) @ eigenvectors.T,
        dtype=np.float64,
    )
    return NoiseModel(
        patch_size=patch_size,
        covariance=covariance,
        whitening=whitening,
        coloring=coloring,
    )


def add_correlated_noise(
    image: FloatImage,
    model: NoiseModel,
    sigma: float,
    rng: np.random.Generator,
) -> FloatImage:
    patch_size = model.patch_size
    height = len(image)
    width = len(image[0])
    noisy = image.copy()
    for row in range(0, height, patch_size):
        for col in range(0, width, patch_size):
            standard_noise = rng.normal(size=patch_size * patch_size)
            patch_noise = sigma * (model.coloring @ standard_noise)
            patch = noisy[row : row + patch_size, col : col + patch_size]
            noisy[row : row + patch_size, col : col + patch_size] = (
                patch + patch_noise.reshape(patch.shape)
            )
    return np.clip(noisy, 0.0, 1.0)


def whiten_patches(image: FloatImage, model: NoiseModel) -> FloatImage:
    patch_size = model.patch_size
    height = len(image)
    width = len(image[0])
    whitened = np.empty_like(image)
    for row in range(0, height, patch_size):
        for col in range(0, width, patch_size):
            patch = image[row : row + patch_size, col : col + patch_size]
            whitened_patch = model.whitening @ patch.reshape(-1)
            whitened[row : row + patch_size, col : col + patch_size] = whitened_patch.reshape(
                patch.shape
            )
    return whitened
