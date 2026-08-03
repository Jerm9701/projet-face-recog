"""Vue PyQt de l’onglet test : panneau de contrôle, vues image, informations et journal."""

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QSplitter
from PyQt5.QtCore import Qt

from views.widgets.control_panel import ControlPanel
from views.widgets.image_view import ImageView
from views.widgets.info_panel import InfoPanel
from views.widgets.log_panel import LogPanel
from tests.test_calibration import TestController


class TestTab(QWidget):

    """Construit l’interface dédiée au test des traitements image."""
    def __init__(self):
        """Initialise l’objet, ses dépendances et l’état interne nécessaire à son fonctionnement."""
        super().__init__()

        self.control_panel = ControlPanel()
        self.input_image_view = ImageView("Image entrée")
        self.result_image_view = ImageView("Image résultat")
        self.info_panel = InfoPanel()
        self.log_panel = LogPanel()

        self.build_layout()

        self.controller = TestController(self)

        self.log_panel.append("Onglet Tests initialisé")

    def build_layout(self):
        """Construit l’agencement des widgets de l’onglet."""
        image_splitter = QSplitter(Qt.Horizontal)
        image_splitter.addWidget(self.input_image_view)
        image_splitter.addWidget(self.result_image_view)
        image_splitter.setSizes([700, 700])

        bottom_splitter = QSplitter(Qt.Horizontal)
        bottom_splitter.addWidget(self.log_panel)
        bottom_splitter.addWidget(self.info_panel)
        bottom_splitter.setSizes([1000, 350])

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.control_panel)
        main_layout.addWidget(image_splitter, stretch=3)
        main_layout.addWidget(bottom_splitter, stretch=1)

        self.setLayout(main_layout)