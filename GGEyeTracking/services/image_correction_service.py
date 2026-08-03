"""Service léger de correction optique, basé sur une calibration chargée."""

import cv2


class ImageCorrectionService:
    """
    Service chargé de corriger les images
    à partir d'une CalibrationResult.
    """

    def __init__(self, calibration_result=None):
        """Initialise l’objet, ses dépendances et l’état interne nécessaire à son fonctionnement."""
        self.calibration = calibration_result

    def set_calibration(self, calibration_result):
        """Injecte un résultat de calibration à utiliser pour les corrections suivantes."""
        self.calibration = calibration_result

    def correct(self, frame):

        """
        Retourne l’image corrigée ou l’image originale si aucune calibration valide n’est
        disponible.
        """
        if self.calibration is None:
            return frame

        if not self.calibration.is_valid():
            return frame

        return cv2.undistort(
            frame,
            self.calibration.camera_matrix,
            self.calibration.distortion_coeffs
        )