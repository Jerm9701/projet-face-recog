"""
Objet de transport utilisé pour faire circuler une image et ses métadonnées dans les
services et pipelines.
"""

from dataclasses import dataclass, field
from typing import Any
import numpy as np
from core.statistics import Statistics

@dataclass
class FrameContext:
    """
    Regroupe l’image source, le résultat, la fonction appliquée, les statistiques et
    d’éventuelles métadonnées.
    """
    input_frame: np.ndarray | None = None
    output_frame: np.ndarray | None = None

    timestamp: float = 0.0
    frame_number: int = 0

    width: int = 0
    height: int = 0

    pipeline_name: str = ""

    faces: list = field(default_factory=list)
    eyes: list = field(default_factory=list)
    gaze: Any = None

    statistics: Statistics = field(default_factory=Statistics)