"""
Vue générique utilisée comme onglet temporaire pour les fonctions non encore
implémentées.
"""

from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QLabel
from PyQt5.QtWidgets import QVBoxLayout
from PyQt5.QtCore import Qt


class PlaceholderTab(QWidget):

    """Onglet temporaire affichant le nom d’une fonction non encore disponible."""
    def __init__(self, name):
        """Initialise l’objet, ses dépendances et l’état interne nécessaire à son fonctionnement."""
        super().__init__()

        label = QLabel(f"{name}\n\nEn cours de développement")
        label.setAlignment(Qt.AlignCenter)

        layout = QVBoxLayout()
        layout.addWidget(label)

        self.setLayout(layout)
        