"""
Fenêtre principale PyQt. Elle assemble la barre d’outils et les onglets de
l’application.
"""

from PyQt5.QtWidgets import (
    QMainWindow,
    QTabWidget,
    QToolBar,
    QAction
)

from views.test_tab import TestTab
from views.calibration_tab import CalibrationTab
from views.placeholder_tab import PlaceholderTab


class MainWindow(QMainWindow):

    """Fenêtre principale qui contient les onglets fonctionnels de l’application."""
    def __init__(self):
        """Initialise l’objet, ses dépendances et l’état interne nécessaire à son fonctionnement."""
        super().__init__()

        self.setWindowTitle("GG Eye Tracking - Test Bench")
        self.resize(1400, 900)

        self.create_toolbar()
        self.create_tabs()
        self.statusBar().showMessage("Ready")

    def create_toolbar(self):
        """Crée la barre d’outils principale."""
        toolbar = QToolBar("Main toolbar")
        self.addToolBar(toolbar)

        open_image_action = QAction("Open image", self)
        open_dataset_action = QAction("Open dataset", self)
        save_action = QAction("Save", self)
        settings_action = QAction("Settings", self)

        toolbar.addAction(open_image_action)
        toolbar.addAction(open_dataset_action)
        toolbar.addSeparator()
        toolbar.addAction(save_action)
        toolbar.addAction(settings_action)

    def create_tabs(self):
        # Chaque onglet regroupe sa vue, puis un contrôleur spécialisé prend en charge sa logique métier.

        """Crée les onglets et associe les contrôleurs nécessaires."""
        self.tabs = QTabWidget()

        self.test_tab = TestTab()
        self.calibration_tab = CalibrationTab()

        self.tabs.addTab(self.test_tab, "Tests")
        self.tabs.addTab(self.calibration_tab, "Calibration")
        self.tabs.addTab(PlaceholderTab("Dataset"), "Dataset")
        self.tabs.addTab(PlaceholderTab("Optimisation"), "Optimisation")
        self.tabs.addTab(PlaceholderTab("Cloud"), "Cloud")
        self.tabs.addTab(PlaceholderTab("Logs"), "Logs")

        self.setCentralWidget(self.tabs)
