"""
Service de sauvegarde des captures. Il crée un dossier de session et enregistre image
source, résultat et métadonnées.
"""

import os
import json
import cv2

from datetime import datetime


class CaptureManager:

    """
    Responsable de la création d’une session de capture et de la sauvegarde cohérente des
    fichiers.
    """
    def __init__(self, root_directory="captures"):
        """Initialise l’objet, ses dépendances et l’état interne nécessaire à son fonctionnement."""
        self.root_directory = root_directory
        os.makedirs(self.root_directory, exist_ok=True)

    def _create_session_directory(self):
        """Crée un dossier unique horodaté pour regrouper une série de captures."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_directory = os.path.join(self.root_directory, timestamp)
        os.makedirs(session_directory, exist_ok=True)
        return session_directory

    def save_snapshot(self, input_frame=None, result_frame=None, metadata=None):
        # Une capture conserve l’image d’entrée, l’image résultat et un fichier JSON décrivant le contexte.

        """Enregistre les images et métadonnées associées à une capture."""
        session_directory = self._create_session_directory()

        if input_frame is not None:
            input_path = os.path.join(session_directory, "input.png")
            cv2.imwrite(input_path, input_frame)

        if result_frame is not None:
            result_path = os.path.join(session_directory, "result.png")
            cv2.imwrite(result_path, result_frame)

        if metadata is None:
            metadata = {}

        metadata["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        metadata_path = os.path.join(session_directory, "metadata.json")

        with open(metadata_path, "w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=4, ensure_ascii=False)

        return session_directory