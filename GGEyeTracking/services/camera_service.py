"""
Service caméra exécuté dans un QThread. Il lit les images OpenCV et émet des signaux Qt
vers l’IHM.
"""

import time
import cv2

from PyQt5.QtCore import QThread, pyqtSignal

from core.frame_context import FrameContext


class CameraService(QThread):

    """Thread Qt chargé de lire la caméra en continu sans bloquer l’interface graphique."""
    frame_ready = pyqtSignal(object)
    status_changed = pyqtSignal(str)

    def __init__(self, camera_index=0):
        """Initialise l’objet, ses dépendances et l’état interne nécessaire à son fonctionnement."""
        super().__init__()

        self.camera_index = camera_index
        self.running = False
        self.capture = None
        self.frame_number = 0
        self.last_time = time.time()

    def run(self):
        # La caméra tourne dans un thread séparé afin que la boucle graphique Qt reste fluide.
        # Chaque frame lue est placée dans un FrameContext avant d’être envoyée au reste de l’application.

        """
        Boucle principale du thread caméra : ouverture, lecture, emballage dans FrameContext et
        émission du signal.
        """
        self.capture = cv2.VideoCapture(self.camera_index)

        if not self.capture.isOpened():
            self.status_changed.emit("Erreur : caméra non ouverte")
            return

        self.status_changed.emit("Caméra ouverte")
        self.running = True

        while self.running:
            start_time = time.time()

            ret, frame = self.capture.read()

            if not ret:
                self.status_changed.emit("Erreur : image non lue")
                break

            self.frame_number += 1

            height, width = frame.shape[:2]

            now = time.time()
            delta_time = now - self.last_time
            fps = 1.0 / delta_time if delta_time > 0 else 0.0
            self.last_time = now

            context = FrameContext(
                input_frame=frame.copy(),
                output_frame=frame.copy(),
                timestamp=now,
                frame_number=self.frame_number,
                width=width,
                height=height
            )

            acquisition_time_ms = (time.time() - start_time) * 1000.0

            context.statistics.timestamp = now
            context.statistics.frame_number = self.frame_number
            context.statistics.acquisition_time_ms = acquisition_time_ms
            context.statistics.fps = fps

            self.frame_ready.emit(context)

            self.msleep(10)

        self.release_camera()

    def stop(self):
        """Arrête le traitement demandé par l’utilisateur."""
        self.running = False

    def release_camera(self):
        """Libère explicitement la caméra OpenCV."""
        if self.capture is not None:
            self.capture.release()
            self.capture = None

        self.status_changed.emit("Caméra fermée")