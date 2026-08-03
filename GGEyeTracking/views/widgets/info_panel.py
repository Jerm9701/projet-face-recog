"""
Widget d’affichage des informations de traitement : modèle, résolution, temps, FPS,
confiance et statut.
"""

from PyQt5.QtWidgets import QGroupBox, QFormLayout, QLabel


class InfoPanel(QGroupBox):

    """Présente les informations synthétiques sur le traitement courant."""
    def __init__(self):
        """Initialise l’objet, ses dépendances et l’état interne nécessaire à son fonctionnement."""
        super().__init__("Informations fonctionnelles")

        self.function_label = QLabel("-")
        self.source_label = QLabel("-")
        self.model_label = QLabel("-")
        self.backend_label = QLabel("-")
        self.resolution_label = QLabel("-")
        self.processing_time_label = QLabel("-")
        self.fps_label = QLabel("-")
        self.confidence_label = QLabel("-")
        self.status_label = QLabel("-")

        layout = QFormLayout()
        layout.addRow("Fonction :", self.function_label)
        layout.addRow("Source :", self.source_label)
        layout.addRow("Modèle :", self.model_label)
        layout.addRow("Backend :", self.backend_label)
        layout.addRow("Résolution :", self.resolution_label)
        layout.addRow("Temps calcul :", self.processing_time_label)
        layout.addRow("FPS :", self.fps_label)
        layout.addRow("Confiance :", self.confidence_label)
        layout.addRow("État :", self.status_label)

        self.setLayout(layout)

    def update_info(
        self,
        function="-",
        source="-",
        model="-",
        backend="-",
        resolution="-",
        processing_time="-",
        fps="-",
        confidence="-",
        status="-"
    ):
        """Met à jour toutes les lignes d’information affichées."""
        self.function_label.setText(function)
        self.source_label.setText(source)
        self.model_label.setText(model)
        self.backend_label.setText(backend)
        self.resolution_label.setText(resolution)
        self.processing_time_label.setText(processing_time)
        self.fps_label.setText(fps)
        self.confidence_label.setText(confidence)
        self.status_label.setText(status)