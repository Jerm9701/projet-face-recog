"""
Vue PyQt de l’onglet calibration : boutons, champs de chemin, zones image, journal et
informations.
"""

from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QLineEdit,
    QFileDialog,
    QSplitter,
)
from PyQt5.QtCore import Qt

from views.widgets.image_view import ImageView
from views.widgets.log_panel import LogPanel
from views.widgets.info_panel import InfoPanel


class CalibrationTab(QWidget):
    """
    Onglet de calibration caméra.

    Il permet :
    - de sélectionner le dossier où seront sauvegardées les images de mire ;
    - de démarrer/arrêter l'acquisition webcam pour créer calib_001.jpg, calib_002.jpg, etc. ;
    - de lancer le calcul de calibration ;
    - de sauvegarder la calibration ;
    - de tester la correction sur une image de mire.
    """

    def __init__(self):
        """Initialise l’objet, ses dépendances et l’état interne nécessaire à son fonctionnement."""
        super().__init__()

        self.directory_edit = QLineEdit("resources/calibration/calib_imgs")

        self.browse_button = QPushButton("Parcourir")
        self.start_capture_button = QPushButton("Start acquisition")
        self.stop_capture_button = QPushButton("Stop acquisition")
        self.calibrate_button = QPushButton("Calibrer")
        self.save_button = QPushButton("Sauvegarder")
        self.test_button = QPushButton("Tester image")

        self.input_view = ImageView("Image mire")
        self.result_view = ImageView("Image corrigée")

        self.info_panel = InfoPanel()
        self.log_panel = LogPanel()

        self._build_layout()

    def _build_layout(self):
        """Assemble les widgets de l’onglet et définit leur organisation visuelle."""
        top_layout = QHBoxLayout()
        top_layout.addWidget(QLabel("Dossier mires :"))
        top_layout.addWidget(self.directory_edit, stretch=1)
        top_layout.addWidget(self.browse_button)
        top_layout.addWidget(self.start_capture_button)
        top_layout.addWidget(self.stop_capture_button)
        top_layout.addWidget(self.calibrate_button)
        top_layout.addWidget(self.save_button)
        top_layout.addWidget(self.test_button)

        image_splitter = QSplitter(Qt.Horizontal)
        image_splitter.addWidget(self.input_view)
        image_splitter.addWidget(self.result_view)
        image_splitter.setSizes([700, 700])

        bottom_splitter = QSplitter(Qt.Horizontal)
        bottom_splitter.addWidget(self.log_panel)
        bottom_splitter.addWidget(self.info_panel)
        bottom_splitter.setSizes([1000, 350])

        main_layout = QVBoxLayout()
        main_layout.addLayout(top_layout)
        main_layout.addWidget(image_splitter, stretch=3)
        main_layout.addWidget(bottom_splitter, stretch=1)

        self.setLayout(main_layout)

    def select_directory(self):
        """Ouvre une boîte de dialogue pour choisir le dossier d’images de calibration."""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Choisir le dossier de mires"
        )

        if directory:
            self.directory_edit.setText(directory)

    def get_directory(self):
        """Retourne le chemin actuellement saisi dans l’interface."""
        return self.directory_edit.text().strip()
