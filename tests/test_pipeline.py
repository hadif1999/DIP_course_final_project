from pathlib import Path

from card.pipeline import run_demo
from card.types import DemoConfig


def test_demo_improves_over_correlated_noisy_input(tmp_path: Path) -> None:
    # Given: a deterministic synthetic correlated-noise restoration experiment.
    config = DemoConfig(image_size=128, output_dir=tmp_path)

    # When: the full demo pipeline runs through its public entry point.
    result = run_demo(config)

    # Then: the CARD-style restoration improves PSNR and writes the figure artifact.
    noisy = result.metrics["Correlated noisy"]
    restored = result.metrics["CARD-style restore"]
    assert restored.psnr > noisy.psnr
    assert Path(result.figure_path).exists()


def test_demo_card_correction_improves_over_gaussian_baseline(tmp_path: Path) -> None:
    # Given: the default experiment with correlated synthetic sensor noise.
    config = DemoConfig(output_dir=tmp_path)

    # When: the full restoration pipeline runs.
    result = run_demo(config)

    # Then: the covariance-aware correction improves the baseline PSNR.
    baseline = result.metrics["Gaussian baseline"]
    restored = result.metrics["CARD-style restore"]
    assert restored.psnr > baseline.psnr
