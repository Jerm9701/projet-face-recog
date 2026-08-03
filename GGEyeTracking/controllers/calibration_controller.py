"""
Contrôleur de l’onglet calibration. Il fait le lien entre les boutons de l’IHM, la
caméra et le service OpenCV de calibration.
"""

import glob
import os

from services.calibration_service import CalibrationService
from services.camera_service import CameraService


class CalibrationController:

    """
    Orchestre le scénario de calibration depuis l’IHM : capture, analyse, calcul, sauvegarde
    et test.
    """
    def __init__(self, tab):
        """Initialise l’objet, ses dépendances et l’état interne nécessaire à son fonctionnement."""
        self.tab = tab
        self.service = CalibrationService()
        self.camera_service = None
        self.last_result = None

        self.connect_signals()

    def connect_signals(self):
        """Connecte les signaux Qt des boutons ou widgets vers les méthodes du contrôleur."""
        self.tab.browse_button.clicked.connect(self.browse)
        self.tab.start_capture_button.clicked.connect(self.start_capture)
        self.tab.stop_capture_button.clicked.connect(self.stop_capture)
        self.tab.calibrate_button.clicked.connect(self.calibrate)
        self.tab.save_button.clicked.connect(self.save)
        self.tab.test_button.clicked.connect(self.test_image)

    def browse(self):
        """Délègue à la vue l’ouverture d’un sélecteur de dossier."""
        self.tab.select_directory()

    def start_capture(self):
        """Prépare une session de capture de mire puis démarre le flux caméra."""
        directory = self.tab.get_directory()

        self.service.start_capture_session(
            directory=directory,
            max_images=25,
            frame_skip=20
        )

        self.tab.log_panel.append("Session de capture calibration démarrée")

        self.camera_service = CameraService(camera_index=0)
        self.camera_service.frame_ready.connect(self.on_frame_ready)
        self.camera_service.status_changed.connect(self.on_camera_status_changed)
        self.camera_service.start()

    def stop_capture(self):
        """Arrête proprement le thread caméra et libère la ressource associée."""
        if self.camera_service is not None:
            self.camera_service.stop()
            self.camera_service.wait()
            self.camera_service = None

        self.tab.log_panel.append("Session de capture calibration arrêtée")

    def on_frame_ready(self, context):
        # Cette méthode est appelée automatiquement par Qt à chaque nouvelle image émise par CameraService.
        # La vue est mise à jour avec l’image annotée et les informations de progression.

        """
        Traite une nouvelle image caméra, met à jour l’affichage et sauvegarde l’image si elle
        est utile.
        """
        found, display, saved_path = self.service.process_capture_frame(
            context.input_frame,
            width=9,
            height=7
        )

        self.tab.input_view.set_image(display)
        self.tab.input_view.set_info(
            f"Images capturées : {self.service.get_capture_count()} / 25"
        )

        if saved_path:
            self.tab.log_panel.append(f"Image sauvegardée : {saved_path}")

        if self.service.is_capture_complete():
            self.tab.log_panel.append("Nombre d'images atteint")
            self.stop_capture()

    def on_camera_status_changed(self, status):
        """Ajoute dans le journal les changements d’état remontés par le service caméra."""
        self.tab.log_panel.append(status)
        

    def calibrate(self):
        """
        Analyse les images de mire, calcule la calibration OpenCV et affiche le résultat dans
        l’IHM.
        """
        directory = self.tab.get_directory()

        try:
            detected_count = 0
            rejected_count = 0

            self.tab.log_panel.append("Vérification des images de calibration...")

            for index, (image_path, original, display, found) in enumerate(
                self.service.analyse_directory(
                    directory=directory,
                    prefix="calib_",
                    image_format="jpg",
                    width=9,
                    height=7
                ),
                start=1
            ):
                self.tab.input_view.set_image(original)
                self.tab.result_view.set_image(display)

                filename = os.path.basename(image_path)

                self.tab.input_view.set_info(f"Image source : {filename}")

                if found:
                    detected_count += 1
                    self.tab.result_view.set_info("Résultat : mire détectée")
                    self.tab.log_panel.append(f"{index:02d} - {filename} : OK")
                else:
                    rejected_count += 1
                    self.tab.result_view.set_info("Résultat : mire non détectée")
                    self.tab.log_panel.append(f"{index:02d} - {filename} : NOK")

            self.tab.log_panel.append(
                f"Vérification terminée : {detected_count} OK / {rejected_count} NOK"
            )

            self.tab.log_panel.append("Calcul de la calibration...")

            self.last_result = self.service.calibrate_from_directory(
                directory=directory,
                prefix="calib_",
                image_format="jpg",
                square_size=0.025,
                width=9,
                height=7
            )

            self.tab.log_panel.append("Calibration réussie")

            self.tab.info_panel.update_info(
                model="OpenCV chessboard",
                resolution=str(self.last_result.image_size),
                processing_time="-",
                fps="-",
                confidence="-",
                status=f"RMS = {self.last_result.rms_error:.4f}"
            )

        except Exception as error:
            self.tab.log_panel.append(f"Erreur calibration : {error}")


    def save(self):
        """Sauvegarde sur disque la dernière calibration valide."""
        try:
            self.service.save("resources/calibration/camera_matrix.yml")
            self.tab.log_panel.append("Calibration sauvegardée")
        except Exception as error:
            self.tab.log_panel.append(f"Erreur sauvegarde : {error}")

    def test_image(self):
        """Charge une image de mire et affiche le résultat corrigé par la calibration courante."""
        directory = self.tab.get_directory()
        images = sorted(glob.glob(os.path.join(directory, "calib_*.jpg")))

        if not images:
            self.tab.log_panel.append("Aucune image de mire trouvée")
            return

        image_path = images[0]

        try:
            import cv2

            original = cv2.imread(image_path)

            if original is None:
                self.tab.log_panel.append("Image impossible à lire")
                return

            corrected = original

            if self.service.result.is_valid():
                corrected = cv2.undistort(
                    original,
                    self.service.result.camera_matrix,
                    self.service.result.distortion_coeffs
                )

            self.tab.input_view.set_image(original)
            self.tab.result_view.set_image(corrected)

            self.tab.input_view.set_info(os.path.basename(image_path))
            self.tab.result_view.set_info("Image corrigée")

            self.tab.log_panel.append("Image de test affichée")

        except Exception as error:
            self.tab.log_panel.append(f"Erreur test image : {error}")