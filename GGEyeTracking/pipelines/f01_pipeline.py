"""
Pipeline F01 : pipeline simple qui renvoie l’image telle quelle tout en mettant à jour
les statistiques.
"""

import time

from pipelines.base_pipeline import BasePipeline


class F01Pipeline(BasePipeline):

    """Pipeline de passage direct utilisé comme référence ou test de fonctionnement."""
    def __init__(self):
        """Initialise l’objet, ses dépendances et l’état interne nécessaire à son fonctionnement."""
        super().__init__("F01 Acquisition")

    def process(self, context):
        """Traite un FrameContext et retourne le contexte enrichi par le pipeline."""
        start_time = time.time()

        context.pipeline_name = self.name

        context.statistics.processing_time_ms = (
            time.time() - start_time
        ) * 1000.0

        return context