"""Widget d’affichage d’image OpenCV dans une interface Qt."""

from PyQt5.QtWidgets import QLabel, QVBoxLayout, QGroupBox
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap

import cv2


class ImageView(QGroupBox):

    """Convertit et affiche des images OpenCV dans un QLabel Qt."""
    def __init__(self, title):
        """Initialise l’objet, ses dépendances et l’état interne nécessaire à son fonctionnement."""
        super().__init__(title)

        self.image_label = QLabel("Aucune image")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(500, 350)
        self.image_label.setStyleSheet(
            "background-color: #202020;"
            "color: white;"
            "border: 1px solid gray;"
        )

        self.info_label = QLabel("-")
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setFixedHeight(24)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label, stretch=1)
        layout.addWidget(self.info_label, stretch=0)

        self.setLayout(layout)

    def set_image(self, frame):
        """Convertit une image OpenCV en QPixmap puis l’affiche."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        height, width, channels = rgb_frame.shape
        bytes_per_line = channels * width

        q_image = QImage(
            rgb_frame.data,
            width,
            height,
            bytes_per_line,
            QImage.Format_RGB888
        )

        pixmap = QPixmap.fromImage(q_image)
        pixmap = pixmap.scaled(
            self.image_label.width(),
            self.image_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        self.image_label.setPixmap(pixmap)

    def set_text(self, text):
        """Affiche un texte simple à la place d’une image."""
        self.image_label.setText(text)

    def set_info(self, text):
        """Met à jour le libellé d’information associé à l’image."""
        self.info_label.setText(text)

    def clear(self):
        """Efface l’image et remet le widget dans un état neutre."""
        self.image_label.clear()
        self.image_label.setText("Aucune image")
        self.info_label.setText("-")