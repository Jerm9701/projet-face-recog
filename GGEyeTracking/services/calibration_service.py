"""
Service métier de calibration caméra. Il détecte une mire, collecte des images, calcule
les paramètres intrinsèques et corrige la distorsion.
"""

import glob
import os

import cv2
import numpy as np

from core.calibration_result import CalibrationResult


class CalibrationService:
    """
    Service dédié à la construction d'une calibration caméra.

    Convention importante : width/height décrivent le nombre de CASES imprimées.
    OpenCV attend le nombre de COINS INTERNES. Une mire 9 x 7 cases donne donc
    un pattern OpenCV de 8 x 6 coins internes.
    """

    def __init__(self):
        """Initialise l’objet, ses dépendances et l’état interne nécessaire à son fonctionnement."""
        self.result = CalibrationResult()

        self.criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001
        )

        self.capture_directory = "resources/calibration/calib_imgs"
        self.capture_count = 0
        self.max_capture_count = 25
        self.frame_skip = 5
        self.frame_counter = 0
        self.last_center = None
        self.min_pose_distance_px = 30.0

    def start_capture_session(
        self,
        directory="resources/calibration/calib_imgs",
        max_images=25,
        frame_skip=5,
        min_pose_distance_px=30.0,
        clear_previous=False
    ):
        """Réinitialise les compteurs et prépare le dossier qui recevra les images de calibration."""
        self.capture_directory = directory
        self.max_capture_count = max_images
        self.frame_skip = max(1, int(frame_skip))
        self.min_pose_distance_px = min_pose_distance_px

        self.capture_count = 0
        self.frame_counter = 0
        self.last_center = None

        os.makedirs(self.capture_directory, exist_ok=True)

        if clear_previous:
            for image_path in glob.glob(os.path.join(self.capture_directory, "calib_*.jpg")):
                os.remove(image_path)

    def process_capture_frame(self, frame, width=9, height=7):
        """
        Analyse une image du flux caméra pour détecter une mire.

        width/height = nombre de cases imprimées.
        Retourne : found, display, saved_path.
        """
        # On travaille sur une copie pour pouvoir dessiner les informations sans modifier l’image source sauvegardée.
        # Le saut de frames évite de lancer une détection OpenCV coûteuse à chaque image caméra.
        # Pour améliorer la calibration, on sauvegarde uniquement les images où la mire a changé suffisamment de position.

        self.frame_counter += 1

        display = frame.copy()
        pattern_size = self._opencv_pattern_size(width, height)

        # Pour ne pas ralentir l'IHM, on ne lance pas OpenCV à chaque frame.
        if self.frame_counter % self.frame_skip != 0:
            self._draw_status(display, False, pattern_size, analyzing=False)
            return False, display, None

        found, refined_corners = self._find_chessboard(frame, pattern_size)

        if not found:
            self._draw_status(display, False, pattern_size, analyzing=True)
            return False, display, None

        cv2.drawChessboardCorners(display, pattern_size, refined_corners, found)

        center = np.mean(refined_corners.reshape(-1, 2), axis=0)

        saved_path = None
        should_save = False

        if self.capture_count < self.max_capture_count:
            if self.last_center is None:
                should_save = True
            else:
                distance = np.linalg.norm(center - self.last_center)
                should_save = distance > self.min_pose_distance_px

        if should_save:
            self.capture_count += 1
            filename = f"calib_{self.capture_count:03d}.jpg"
            saved_path = os.path.join(self.capture_directory, filename)
            cv2.imwrite(saved_path, frame)
            self.last_center = center

        self._draw_status(display, True, pattern_size, analyzing=True)
        return True, display, saved_path

    def get_capture_count(self):
        """Retourne le nombre d’images de calibration déjà sauvegardées."""
        return self.capture_count

    def is_capture_complete(self):
        """Indique si le nombre cible d’images de calibration est atteint."""
        return self.capture_count >= self.max_capture_count

    def analyse_directory(
        self,
        directory,
        prefix="calib_",
        image_format="jpg",
        width=9,
        height=7
    ):
        """
        Analyse les images de calibration une par une.
    
        Retourne un générateur produisant :
            image_path,
            original,
            display,
            found
        """
        directory = directory.rstrip("/\\")
        images = sorted(glob.glob(os.path.join(directory, f"{prefix}*.{image_format}")))
    
        pattern_size = self._opencv_pattern_size(width, height)
    
        for image_path in images:
        
            original = cv2.imread(image_path)
    
            if original is None:
                continue
            
            found, display = self.detect_chessboard(
                original,
                width,
                height
            )
    
            yield image_path, original, display, found

            
    def calibrate_from_directory(
        self,
        directory,
        prefix="calib_",
        image_format="jpg",
        square_size=0.025,
        width=9,
        height=7
    ):
        # Préparation des listes OpenCV : objpoints contient les points 3D théoriques de la mire, imgpoints les points 2D détectés dans les images.
        # La première mire détectée sert à confirmer automatiquement la taille réelle du pattern utilisé par OpenCV.
        # cv2.calibrateCamera calcule la matrice intrinsèque, les coefficients de distorsion et l’erreur RMS globale.

        """
        Construit les points 2D/3D à partir des images détectées puis lance
        cv2.calibrateCamera().
        """
        objpoints = []
        imgpoints = []

        directory = directory.rstrip("/\\")
        images = sorted(glob.glob(os.path.join(directory, f"{prefix}*.{image_format}")))

        if not images:
            raise FileNotFoundError(f"Aucune image trouvée dans : {directory}")

        detected_pattern_size = None
        object_points_template = None
        image_size = None

        for image_path in images:
            image = cv2.imread(image_path)
            if image is None:
                continue

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image_size = gray.shape[::-1]

            if detected_pattern_size is None:
                detected_pattern_size, object_points_template = self._detect_pattern_size(
                    gray, width, height, square_size
                )
                if detected_pattern_size is None:
                    continue

            found, refined_corners = self._find_chessboard(image, detected_pattern_size)

            if found:
                objpoints.append(object_points_template)
                imgpoints.append(refined_corners)

        if not imgpoints:
            raise ValueError(
                "Aucun coin de mire détecté. Vérifier que width/height décrivent "
                "le nombre de cases imprimées. Pour une mire 9x7 cases, OpenCV utilise 8x6 coins internes."
            )

        rms, camera_matrix, distortion_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            objpoints,
            imgpoints,
            image_size,
            None,
            None
        )

        self.result = CalibrationResult(
            camera_matrix=camera_matrix,
            distortion_coeffs=distortion_coeffs,
            rms_error=rms,
            image_size=image_size,
            pattern_size=detected_pattern_size
        )

        return self.result

    def _opencv_pattern_size(self, width, height):
        """Convertit un nombre de cases imprimées en nombre de coins internes OpenCV."""
        return max(1, int(width) - 1), max(1, int(height) - 1)

    def _find_chessboard(self, frame, pattern_size):
        """Cherche les coins de la mire et affine leur position au sous-pixel."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        flags = (
            cv2.CALIB_CB_ADAPTIVE_THRESH
            + cv2.CALIB_CB_NORMALIZE_IMAGE
            + cv2.CALIB_CB_FAST_CHECK
        )

        found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)

        # Fallback plus robuste pour certaines webcams / mires imprimées.
        if not found and hasattr(cv2, "findChessboardCornersSB"):
            found, corners = cv2.findChessboardCornersSB(gray, pattern_size, None)

        if not found:
            return False, None

        refined_corners = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            self.criteria
        )

        return True, refined_corners

    def _detect_pattern_size(self, gray, width, height, square_size):
        # Priorité au cas normal : width/height = cases imprimées.
        """Teste plusieurs tailles de mire possibles afin de rendre la calibration plus tolérante."""
        inner = self._opencv_pattern_size(width, height)
        candidate_sizes = [
            inner,
            (width, height),
            (width - 1, height),
            (width, height - 1),
        ]

        for pattern_size in candidate_sizes:
            if pattern_size[0] <= 0 or pattern_size[1] <= 0:
                continue

            # _find_chessboard attend une image BGR ; ici on a déjà gray.
            flags = (
                cv2.CALIB_CB_ADAPTIVE_THRESH
                + cv2.CALIB_CB_NORMALIZE_IMAGE
                + cv2.CALIB_CB_FAST_CHECK
            )
            found, _ = cv2.findChessboardCorners(gray, pattern_size, flags)
            if not found and hasattr(cv2, "findChessboardCornersSB"):
                found, _ = cv2.findChessboardCornersSB(gray, pattern_size, None)

            if found:
                objp = np.zeros((pattern_size[1] * pattern_size[0], 3), np.float32)
                objp[:, :2] = np.mgrid[
                    0:pattern_size[0],
                    0:pattern_size[1]
                ].T.reshape(-1, 2)
                objp *= square_size
                return pattern_size, objp

        return None, None

    def _draw_status(self, display, found, pattern_size, analyzing=True):
        """Dessine directement sur l’image un statut lisible par l’utilisateur."""
        color = (0, 255, 0) if found else (0, 0, 255)
        label = "Mire detectee" if found else "Mire non detectee"
        if not analyzing:
            label = "Attente analyse"

        text = f"{label} - {self.capture_count}/{self.max_capture_count} - pattern {pattern_size[0]}x{pattern_size[1]}"
        cv2.putText(display, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    def save(self, path):
        """Sauvegarde sur disque la dernière calibration valide."""
        if not self.result.is_valid():
            raise RuntimeError("Aucune calibration valide à sauvegarder.")

        os.makedirs(os.path.dirname(path), exist_ok=True)

        cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
        cv_file.write("K", self.result.camera_matrix)
        cv_file.write("D", self.result.distortion_coeffs)
        cv_file.write("rms", self.result.rms_error)
        cv_file.write("image_width", self.result.image_size[0])
        cv_file.write("image_height", self.result.image_size[1])
        cv_file.write("pattern_width", self.result.pattern_size[0])
        cv_file.write("pattern_height", self.result.pattern_size[1])
        cv_file.release()

    def load(self, path="resources/calibration/camera_matrix.yml"):
        """Recharge une calibration sauvegardée dans un fichier YAML OpenCV."""
        if not os.path.exists(path):
            return None

        cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)

        camera_matrix = cv_file.getNode("K").mat()
        distortion_coeffs = cv_file.getNode("D").mat()
        rms_error = cv_file.getNode("rms").real()

        image_width = int(cv_file.getNode("image_width").real())
        image_height = int(cv_file.getNode("image_height").real())
        pattern_width = int(cv_file.getNode("pattern_width").real())
        pattern_height = int(cv_file.getNode("pattern_height").real())

        cv_file.release()

        self.result = CalibrationResult(
            camera_matrix=camera_matrix,
            distortion_coeffs=distortion_coeffs,
            rms_error=rms_error,
            image_size=(image_width, image_height),
            pattern_size=(pattern_width, pattern_height)
        )

        return self.result

    def detect_chessboard(self, frame, width=9, height=7):
        """Détecte une mire sur une image isolée et renvoie une image annotée."""
        pattern_size = self._opencv_pattern_size(width, height)
        found, refined_corners = self._find_chessboard(frame, pattern_size)

        if not found:
            return False, frame

        display = frame.copy()
        cv2.drawChessboardCorners(display, pattern_size, refined_corners, found)
        return True, display

    def undistort(self, frame):
        """Corrige la distorsion d’une image si une calibration valide est disponible."""
        if not self.result.is_valid():
            return frame

        return cv2.undistort(
            frame,
            self.result.camera_matrix,
            self.result.distortion_coeffs
        )
