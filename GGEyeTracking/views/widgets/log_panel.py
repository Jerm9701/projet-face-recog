"""Widget de journalisation simple avec horodatage des messages."""

from datetime import datetime

from PyQt5.QtWidgets import QGroupBox, QVBoxLayout, QTextEdit


class LogPanel(QGroupBox):

    """Affiche les messages de suivi avec un horodatage."""
    def __init__(self):
        """Initialise l’objet, ses dépendances et l’état interne nécessaire à son fonctionnement."""
        super().__init__("Console / Logs")

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)

        layout = QVBoxLayout()
        layout.addWidget(self.log_box)

        self.setLayout(layout)

    def append(self, message):
        """Ajoute un message horodaté au journal."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_box.append(f"{timestamp}  {message}")

    def clear(self):
        """Efface l’image et remet le widget dans un état neutre."""
        self.log_box.clear()