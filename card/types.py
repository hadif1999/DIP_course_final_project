from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

FloatImage = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class DemoConfig:
    image_size: int = 256
    patch_size: int = 8
    noise_sigma: float = 0.12
    correlation_strength: float = 0.92
    regularization: float = 1e-5
    seed: int = 7
    output_dir: Path = Path("results")
