"""Emplacement prévu pour un futur pipeline F03."""

from pipelines.base_pipeline import BasePipeline


class F03Pipeline(BasePipeline):

    """Pipeline réservé à une future fonction F03."""
    def __init__(self):
        """Initialise l’objet, ses dépendances et l’état interne nécessaire à son fonctionnement."""
        super().__init__("F03 Face Detection")

    def process(self, frame):
        """Traite un FrameContext et retourne le contexte enrichi par le pipeline."""
        return frame