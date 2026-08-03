"""
Contrôleur global de l’application. Il relie la fenêtre principale aux contrôleurs
spécialisés de chaque onglet.
"""

from controllers.calibration_controller import CalibrationController


class ApplicationController:

    """Coordonne les contrôleurs de haut niveau associés à la fenêtre principale."""
    def __init__(self, main_window):
        """Initialise l’objet, ses dépendances et l’état interne nécessaire à son fonctionnement."""
        self.main_window = main_window
        self.current_tab_index = 0

        self.calibration_controller = CalibrationController(
            self.main_window.calibration_tab
        )

        self.main_window.tabs.currentChanged.connect(self.on_tab_changed)

    def on_tab_changed(self, index):
        """Réagit au changement d’onglet pour adapter le comportement global si nécessaire."""
        self.current_tab_index = index
        tab_name = self.main_window.tabs.tabText(index)
        self.main_window.statusBar().showMessage(f"Onglet actif : {tab_name}")