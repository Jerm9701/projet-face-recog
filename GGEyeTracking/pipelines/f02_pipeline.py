"""
Pipeline F02 : exemple de traitement OpenCV convertissant l’image en niveaux de gris
puis en BGR pour l’affichage.
"""

import time

from pipelines.base_pipeline import BasePipeline
from services.calibration_service import CalibrationService
from services.image_correction_service import ImageCorrectionService


class F02Pipeline(BasePipeline):

    """Pipeline de démonstration appliquant une conversion en niveaux de gris."""
    def __init__(self):
        """Initialise l’objet, ses dépendances et l’état interne nécessaire à son fonctionnement."""
        super().__init__("F02 Correction des iamges")

        self.calibration_service = CalibrationService()
        self.calibration_result = self.calibration_service.load()

        self.correction_service = ImageCorrectionService(
            self.calibration_result
        )

    def process(self, context):

        """Traite un FrameContext et retourne le contexte enrichi par le pipeline."""
        start = time.perf_counter()

        context.pipeline_name = self.name

        if self.calibration_result is not None and self.calibration_result.is_valid():

            context.output_frame = self.correction_service.correct(
                context.input_frame
            )

            context.statistics.confidence = 1.0
            context.statistics.calibrated = True

        else:

            context.output_frame = context.input_frame
            context.statistics.confidence = 0.0
            context.statistics.calibrated = False

        context.statistics.processing_time_ms = (
            time.perf_counter() - start
        ) * 1000

        return context