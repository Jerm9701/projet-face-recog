"""
Gestionnaire de pipelines. Il choisit le pipeline demandé par l’IHM et lui délègue le
traitement des images.
"""

from pipelines.f01_pipeline import F01Pipeline
from pipelines.f02_pipeline import F02Pipeline


class PipelineManager:

    """Instancie et exécute le pipeline correspondant à la fonction sélectionnée."""
    def __init__(self):
        """Initialise l’objet, ses dépendances et l’état interne nécessaire à son fonctionnement."""
        self.current_pipeline = F01Pipeline()

    def configure(self, function_name):
        """Sélectionne le pipeline correspondant à la fonction demandée."""
        if function_name.startswith("F01"):
            self.current_pipeline = F01Pipeline()

        elif function_name.startswith("F02"):
            self.current_pipeline = F02Pipeline()

        else:
            self.current_pipeline = F01Pipeline()

    def start(self):
        """Démarre le traitement demandé par l’utilisateur."""
        self.current_pipeline.start()
        return {
            "status": "Démarré",
            "message": f"Pipeline démarré : {self.current_pipeline.name}"
        }

    def stop(self):
        """Arrête le traitement demandé par l’utilisateur."""
        self.current_pipeline.stop()
        return {
            "status": "Arrêté",
            "message": "Pipeline arrêté"
        }

    def process(self, context):
        """Traite un FrameContext et retourne le contexte enrichi par le pipeline."""
        return self.current_pipeline.process(context)