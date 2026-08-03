"""Emplacement prévu pour un futur pipeline F04."""

class F04Pipeline(BasePipeline):

    """Pipeline réservé à une future fonction F04."""
    def __init__(self):
        """Initialise l’objet, ses dépendances et l’état interne nécessaire à son fonctionnement."""
        super().__init__("F04 Eye Segmentation")
        # self.segmenter = EyeSegmenter()

    def process(self, frame):

        """Traite un FrameContext et retourne le contexte enrichi par le pipeline."""
        eyes = self.segmenter.segment(frame)

        return self.segmenter.draw(frame, eyes)