import numpy as np

from card.noise import build_noise_model, make_rolling_shutter_covariance, whiten_patches


def test_whitening_transform_makes_covariance_identity() -> None:
    # Given: a strongly correlated rolling-shutter covariance model.
    covariance = make_rolling_shutter_covariance(8, 0.9, 1e-5)
    model = build_noise_model(covariance)

    # When: the whitening matrix is applied to the covariance.
    whitened_covariance = model.whitening @ covariance @ model.whitening.T

    # Then: the covariance is approximately identity.
    np.testing.assert_allclose(whitened_covariance, np.eye(64), atol=1e-7)


def test_whiten_patches_preserves_image_shape() -> None:
    # Given: an image with dimensions divisible by the patch size.
    image = np.ones((16, 16), dtype=np.float64)
    covariance = make_rolling_shutter_covariance(8, 0.8, 1e-5)
    model = build_noise_model(covariance)

    # When: whitening is applied patchwise.
    whitened = whiten_patches(image, model)

    # Then: the operation keeps the image layout intact.
    assert whitened.shape == image.shape
