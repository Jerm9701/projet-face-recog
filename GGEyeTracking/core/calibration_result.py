"""Objet de données contenant le résultat mathématique d’une calibration caméra OpenCV."""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class CalibrationResult:
    """
    Structure le résultat de calibration : matrice caméra, coefficients de distorsion,
    erreur RMS et dimensions.
    """
    camera_matrix: Optional[np.ndarray] = None
    distortion_coeffs: Optional[np.ndarray] = None
    rms_error: float = 0.0
    image_size: Tuple[int, int] = (0, 0)
    pattern_size: Tuple[int, int] = (0, 0)

    def is_valid(self):
        """Vérifie que les données minimales de calibration sont présentes."""
        return (
            self.camera_matrix is not None
            and self.distortion_coeffs is not None
        )