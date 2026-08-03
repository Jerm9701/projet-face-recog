"""
Widget de contrôle permettant de choisir une fonction et une source vidéo puis de
lancer/arrêter les traitements.
"""

from PyQt5.QtWidgets import (
    QGroupBox, QHBoxLayout, QLabel, QComboBox, QPushButton
)
from PyQt5.QtCore import pyqtSignal


class ControlPanel(QGroupBox):

    """Panneau utilisateur contenant les sélections et boutons d’action."""
    start_clicked = pyqtSignal()
    stop_clicked = pyqtSignal()
    snapshot_clicked = pyqtSignal()
    batch_clicked = pyqtSignal()

    def __init__(self):
        """Initialise l’objet, ses dépendances et l’état interne nécessaire à son fonctionnement."""
        super().__init__("Commandes")

        self.function_selector = QComboBox()
        self.function_selector.addItems([
            "F01 - Acquisition vidéo",
            "F02 - Calibration caméra",
            "F03 - Détection de visage",
            "F04 - Segmentation des yeux",
            "F05 - Calcul trajectoire regard",
            "F06 - Optimisation",
            "F07 - Cloud / AWS simulation"
        ])

        self.source_selector = QComboBox()
        self.source_selector.addItems([
            "Webcam",
            "Image",
            "Dataset"
        ])

        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")
        self.snapshot_button = QPushButton("Snapshot")
        self.batch_button = QPushButton("Batch")

        layout = QHBoxLayout()
        layout.addWidget(QLabel("Fonction :"))
        layout.addWidget(self.function_selector)
        layout.addWidget(QLabel("Source :"))
        layout.addWidget(self.source_selector)
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)
        layout.addWidget(self.snapshot_button)
        layout.addWidget(self.batch_button)

        self.setLayout(layout)

        self.start_button.clicked.connect(self.start_clicked.emit)
        self.stop_button.clicked.connect(self.stop_clicked.emit)
        self.snapshot_button.clicked.connect(self.snapshot_clicked.emit)
        self.batch_button.clicked.connect(self.batch_clicked.emit)

    def get_selected_function(self):
        """Retourne la fonction choisie dans la liste déroulante."""
        return self.function_selector.currentText()

    def get_selected_source(self):
        """Retourne la source vidéo choisie dans la liste déroulante."""
        return self.source_selector.currentText()