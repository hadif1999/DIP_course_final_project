import numpy as np

from card.noise import NoiseModel
from card.types import FloatImage

CARD_CORRECTION_WEIGHT = 0.3


def gaussian_baseline(noisy: FloatImage, sigma: float = 0.9) -> FloatImage:
    restored = _separable_gaussian_blur(noisy, sigma)
    return np.asarray(np.clip(restored, 0.0, 1.0), dtype=np.float64)


def card_wiener_restore(noisy: FloatImage, model: NoiseModel, noise_sigma: float) -> FloatImage:
    patch_size = model.patch_size
    eigenvalues, eigenvectors = np.linalg.eigh(model.covariance)
    patches = _patch_matrix(noisy, patch_size)
    coeffs = eigenvectors.T @ patches
    observed_var = np.var(coeffs, axis=1, ddof=1)
    noise_var = (noise_sigma**2) * np.maximum(eigenvalues, 1e-10)
    signal_var = np.maximum(observed_var - noise_var, 0.0)
    gains = signal_var / (signal_var + noise_var + 1e-10)
    restored_patches = eigenvectors @ (gains[:, None] * coeffs)
    shape = (len(noisy), len(noisy[0]))
    covariance_restored = _image_from_patches(restored_patches, shape, patch_size)
    image_prior = gaussian_baseline(noisy)
    restored = (
        CARD_CORRECTION_WEIGHT * covariance_restored
        + (1.0 - CARD_CORRECTION_WEIGHT) * image_prior
    )
    return np.asarray(np.clip(restored, 0.0, 1.0), dtype=np.float64)


def _patch_matrix(image: FloatImage, patch_size: int) -> FloatImage:
    height = len(image)
    width = len(image[0])
    patch_count = (height // patch_size) * (width // patch_size)
    patches = np.empty((patch_size * patch_size, patch_count), dtype=np.float64)
    index = 0
    for row in range(0, height, patch_size):
        for col in range(0, width, patch_size):
            patch = image[row : row + patch_size, col : col + patch_size]
            patches[:, index] = patch.reshape(-1)
            index += 1
    return patches


def _separable_gaussian_blur(image: FloatImage, sigma: float) -> FloatImage:
    radius = max(1, round(3 * sigma))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(offsets**2) / (2 * sigma**2))
    kernel = np.asarray(kernel / float(np.sum(kernel)), dtype=np.float64)
    horizontal = _convolve_rows(image, kernel, radius)
    return _convolve_columns(horizontal, kernel, radius)


def _convolve_rows(image: FloatImage, kernel: FloatImage, radius: int) -> FloatImage:
    output = np.empty_like(image)
    for row_index, row in enumerate(image):
        padded = np.pad(row, radius, mode="reflect")
        output[row_index, :] = np.convolve(padded, kernel, mode="valid")
    return output


def _convolve_columns(image: FloatImage, kernel: FloatImage, radius: int) -> FloatImage:
    output = np.empty_like(image)
    for col_index in range(len(image[0])):
        column = image[:, col_index]
        padded = np.pad(column, radius, mode="reflect")
        output[:, col_index] = np.convolve(padded, kernel, mode="valid")
    return output


def _image_from_patches(patches: FloatImage, shape: tuple[int, int], patch_size: int) -> FloatImage:
    image = np.empty(shape, dtype=np.float64)
    index = 0
    for row in range(0, shape[0], patch_size):
        for col in range(0, shape[1], patch_size):
            image[row : row + patch_size, col : col + patch_size] = patches[:, index].reshape(
                patch_size,
                patch_size,
            )
            index += 1
    return image
