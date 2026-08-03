"""
Point d’entrée de l’application PyQt. Il crée QApplication, instancie la fenêtre
principale puis lance la boucle événementielle.
"""

import sys

from PyQt5.QtWidgets import QApplication

from views.main_window import MainWindow
from controllers.application_controller import ApplicationController


def main():
    """Méthode main : voir son nom et ses appels pour le rôle exact."""
    app = QApplication(sys.argv)

    window = MainWindow()
    controller = ApplicationController(window)

    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()