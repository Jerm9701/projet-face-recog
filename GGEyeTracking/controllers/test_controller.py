"""
Contrôleur de l’onglet de test. Il démarre le flux caméra, applique le pipeline
sélectionné et gère les captures.
"""

from services.pipeline_manager import PipelineManager
from services.camera_service import CameraService
from services.capture_manager import CaptureManager


class TestController:

    """Pilote le mode test : caméra, pipeline courant, affichage des résultats et captures."""
    def __init__(self, test_tab):
        """Initialise l’objet, ses dépendances et l’état interne nécessaire à son fonctionnement."""
        self.test_tab = test_tab

        self.pipeline_manager = PipelineManager()
        self.camera_service = None
        self.capture_manager = CaptureManager()

        self.last_context = None
        self.last_input_frame = None
        self.last_result_frame = None

        self.connect_signals()

    def connect_signals(self):
        """Connecte les signaux Qt des boutons ou widgets vers les méthodes du contrôleur."""
        self.test_tab.control_panel.start_clicked.connect(self.start)
        self.test_tab.control_panel.stop_clicked.connect(self.stop)
        self.test_tab.control_panel.snapshot_clicked.connect(self.snapshot)
        self.test_tab.control_panel.batch_clicked.connect(self.batch)

    def start(self):
        """Démarre le traitement demandé par l’utilisateur."""
        function_name = self.test_tab.control_panel.get_selected_function()
        source_name = self.test_tab.control_panel.get_selected_source()

        self.pipeline_manager.configure(function_name)
        result = self.pipeline_manager.start()

        self.test_tab.info_panel.update_info(
            function=function_name,
            source=source_name,
            status=result["status"]
        )

        self.test_tab.log_panel.append(result["message"])

        if source_name == "Webcam":
            self.start_camera()
        else:
            self.test_tab.log_panel.append("Source non encore implémentée")

    def start_camera(self):
        """Configure et démarre la caméra pour alimenter le pipeline en images."""
        if self.camera_service is not None:
            self.stop_camera()

        self.camera_service = CameraService(camera_index=0)
        self.camera_service.frame_ready.connect(self.on_frame_ready)
        self.camera_service.status_changed.connect(self.on_camera_status_changed)
        self.camera_service.start()

    def stop(self):
        """Arrête le traitement demandé par l’utilisateur."""
        self.stop_camera()

        result = self.pipeline_manager.stop()

        self.test_tab.input_image_view.clear()
        self.test_tab.result_image_view.clear()

        self.test_tab.info_panel.update_info(status=result["status"])
        self.test_tab.log_panel.append(result["message"])

    def stop_camera(self):
        """Stoppe la caméra et attend la fin du thread pour éviter les accès concurrents."""
        if self.camera_service is not None:
            self.camera_service.stop()
            self.camera_service.wait()
            self.camera_service = None

    def on_frame_ready(self, context):
        # Cette méthode est appelée automatiquement par Qt à chaque nouvelle image émise par CameraService.
        # La vue est mise à jour avec l’image annotée et les informations de progression.

        """
        Traite une nouvelle image caméra, met à jour l’affichage et sauvegarde l’image si elle
        est utile.
        """
        context = self.pipeline_manager.process(context)

        self.last_context = context
        self.last_input_frame = context.input_frame.copy()
        self.last_result_frame = context.output_frame.copy()

        stats = context.statistics

        self.test_tab.input_image_view.set_image(context.input_frame)
        self.test_tab.input_image_view.set_info(
            f"Source : Webcam | Résolution : {context.width}x{context.height} | FPS : {stats.fps:.1f}"
        )

        self.test_tab.result_image_view.set_image(context.output_frame)
        self.test_tab.result_image_view.set_info(
            f"Pipeline : {context.pipeline_name} | Temps : {stats.processing_time_ms:.1f} ms"
        )

        self.test_tab.info_panel.update_info(
            resolution=f"{context.width}x{context.height}",
            processing_time=f"{stats.processing_time_ms:.1f} ms",
            fps=f"{stats.fps:.1f}",
            confidence=f"{stats.confidence * 100:.1f} %",
            status="Caméra active"
        )

    def on_camera_status_changed(self, status):
        """Ajoute dans le journal les changements d’état remontés par le service caméra."""
        self.test_tab.log_panel.append(status)

    def snapshot(self):
        """Sauvegarde une capture du contexte courant."""
        if self.last_input_frame is None:
            self.test_tab.log_panel.append("Snapshot impossible : aucune image disponible")
            return

        function_name = self.test_tab.control_panel.get_selected_function()
        source_name = self.test_tab.control_panel.get_selected_source()

        metadata = {
            "function": function_name,
            "source": source_name,
            "status": "snapshot",
            "frame_number": self.last_context.frame_number if self.last_context else None,
            "fps": self.last_context.statistics.fps if self.last_context else None,
            "processing_time_ms": self.last_context.statistics.processing_time_ms if self.last_context else None,
        }

        directory = self.capture_manager.save_snapshot(
            input_frame=self.last_input_frame,
            result_frame=self.last_result_frame,
            metadata=metadata
        )

        self.test_tab.log_panel.append(f"Snapshot sauvegardé : {directory}")

    def batch(self):
        """Point d’entrée prévu pour un traitement par lot."""
        self.test_tab.log_panel.append("Batch test demandé")