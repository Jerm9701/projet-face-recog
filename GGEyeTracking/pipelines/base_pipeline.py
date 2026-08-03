"""
Classe de base commune aux pipelines de traitement image. Elle définit le contrat
minimal start/stop/process/statistics.
"""

from abc import ABC, abstractmethod


class BasePipeline(ABC):
    """
    Classe de base de tous les pipelines F01 à F05.
    """

    def __init__(self, name):
        """Initialise l’objet, ses dépendances et l’état interne nécessaire à son fonctionnement."""
        self.name = name
        self.running = False

    def start(self):
        """Démarre le traitement demandé par l’utilisateur."""
        self.running = True

    def stop(self):
        """Arrête le traitement demandé par l’utilisateur."""
        self.running = False

    @abstractmethod
    def process(self, frame):
        """
        Traite une image.
        Retourne l'image résultat.
        """
        pass

    def get_statistics(self):
        """
        Retourne les statistiques du pipeline.
        """
        return {
            "model": "-",
            "backend": "-",
            "fps": "-",
            "processing_time": "-",
            "confidence": "-"
        }