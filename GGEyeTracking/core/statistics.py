"""
Objet de données regroupant les indicateurs de performance et d’état affichés dans
l’IHM.
"""

from dataclasses import dataclass


@dataclass
class Statistics:
    """Regroupe les indicateurs numériques et textuels affichés dans l’interface."""
    timestamp: float = 0.0
    frame_number: int = 0

    acquisition_time_ms: float = 0.0
    processing_time_ms: float = 0.0
    display_time_ms: float = 0.0
    total_time_ms: float = 0.0

    fps: float = 0.0
    confidence: float = 0.0

    nb_faces: int = 0
    nb_eyes: int = 0

    cpu_percent: float = 0.0
    gpu_percent: float = 0.0
    ram_percent: float = 0.0
    gpu_memory_mb: float = 0.0