#!/usr/bin/env python3
"""
Test an ArcFace/IResNet checkpoint.

Mode 1:
    python test_recognition.py folder --checkpoint runs\casia_arcface\best.pt --input-dir 

Mode 2:
    python test_recognition.py camera --checkpoint runs\casia_arcface\best.pt --camera-id 0

Use best.pt or last.pt. backbone_final.pt does not contain the class weights.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from model import iresnet50

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class FaceClassifier:
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "auto",
        unknown_threshold: float = 0.0,
    ) -> None:
        self.device = self._resolve_device(device)
        self.unknown_threshold = unknown_threshold

        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        for key in ("backbone", "head", "class_to_idx"):
            if key not in checkpoint:
                raise ValueError(
                    f"Checkpoint is missing '{key}'. Use best.pt or last.pt."
                )

        config = checkpoint.get("config", {})
        embedding_dim = int(config.get("embedding_dim", 512))
        dropout = float(config.get("dropout", 0.4))
        self.arcface_scale = float(config.get("arcface_scale", 64.0))

        self.backbone = iresnet50(
            embedding_dim=embedding_dim,
            dropout=dropout,
        )
        self.backbone.load_state_dict(checkpoint["backbone"])
        self.backbone.to(self.device).eval()

        head_state = checkpoint["head"]
        if "weight" not in head_state:
            raise ValueError("ArcFace head state has no weight tensor.")

        self.class_weights = F.normalize(
            head_state["weight"].float().to(self.device), dim=1
        )

        class_to_idx: Dict[str, int] = checkpoint["class_to_idx"]
        self.idx_to_class = {
            int(index): class_name for class_name, index in class_to_idx.items()
        }

        paper_std = 128.0 / 255.0
        self.transform = transforms.Compose(
            [
                transforms.Resize((112, 112)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.5, 0.5, 0.5),
                    std=(paper_std, paper_std, paper_std),
                ),
            ]
        )

        print(
            f"Loaded {len(self.idx_to_class)} classes on {self.device}; "
            f"unknown threshold={self.unknown_threshold:.3f}"
        )

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        requested = torch.device(device)
        if requested.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return requested

    @torch.inference_mode()
    def predict_rgb(
        self, rgb_image: np.ndarray, top_k: int = 1
    ) -> List[Tuple[str, float, float]]:
        """Return (class name, softmax confidence, cosine similarity)."""
        if rgb_image.size == 0:
            raise ValueError("Empty image received.")

        tensor = self.transform(Image.fromarray(rgb_image)).unsqueeze(0)
        tensor = tensor.to(self.device)

        embedding = F.normalize(self.backbone(tensor), dim=1)
        cosine_logits = F.linear(embedding, self.class_weights)
        probabilities = F.softmax(cosine_logits * self.arcface_scale, dim=1)

        k = min(top_k, probabilities.shape[1])
        confidences, indices = probabilities.topk(k, dim=1)

        results: List[Tuple[str, float, float]] = []
        for confidence, index in zip(confidences[0], indices[0]):
            class_index = int(index.item())
            cosine = float(cosine_logits[0, class_index].item())
            class_name = self.idx_to_class.get(class_index, f"class_{class_index}")
            if cosine < self.unknown_threshold:
                class_name = "UNKNOWN"
            results.append((class_name, float(confidence.item()), cosine))
        return results


def iter_images(input_dir: Path, recursive: bool) -> Iterable[Path]:
    iterator = input_dir.rglob("*") if recursive else input_dir.glob("*")
    for path in sorted(iterator):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def load_bgr_image(path: Path) -> Optional[np.ndarray]:
    raw = np.fromfile(str(path), dtype=np.uint8)
    if raw.size == 0:
        return None
    return cv2.imdecode(raw, cv2.IMREAD_COLOR)


def run_folder_mode(args: argparse.Namespace) -> None:
    classifier = FaceClassifier(
        args.checkpoint, args.device, args.unknown_threshold
    )
    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Invalid input folder: {input_dir}")

    image_paths = list(iter_images(input_dir, args.recursive))
    if not image_paths:
        print(f"No supported images found in {input_dir}")
        return

    rows = []
    for image_path in image_paths:
        bgr = load_bgr_image(image_path)
        if bgr is None:
            print(f"[ERROR] Cannot read: {image_path}")
            continue

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        predictions = classifier.predict_rgb(rgb, top_k=args.top_k)
        best_class, best_confidence, best_cosine = predictions[0]

        print(
            f"{image_path.name}: class={best_class} "
            f"confidence={best_confidence:.4f} cosine={best_cosine:.4f}"
        )

        if args.top_k > 1:
            for rank, (name, confidence, cosine) in enumerate(predictions, 1):
                print(
                    f"  {rank}. {name}: confidence={confidence:.4f}, "
                    f"cosine={cosine:.4f}"
                )

        rows.append(
            {
                "image": str(image_path),
                "predicted_class": best_class,
                "confidence": best_confidence,
                "cosine_similarity": best_cosine,
            }
        )

    if args.output_csv:
        output_csv = Path(args.output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"Results saved to: {output_csv}")


def create_face_detector() -> cv2.CascadeClassifier:
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        raise RuntimeError(f"Cannot load Haar cascade: {cascade_path}")
    return detector


def run_camera_mode(args: argparse.Namespace) -> None:
    classifier = FaceClassifier(
        args.checkpoint, args.device, args.unknown_threshold
    )
    detector = create_face_detector()

    capture = cv2.VideoCapture(args.camera_id)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open camera {args.camera_id}")

    if args.width > 0:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height > 0:
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    print("Press Q or ESC to stop.")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("Cannot read camera frame.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(
                gray,
                scaleFactor=args.scale_factor,
                minNeighbors=args.min_neighbors,
                minSize=(args.min_face_size, args.min_face_size),
            )

            for x, y, width, height in faces:
                margin_x = int(width * args.face_margin)
                margin_y = int(height * args.face_margin)
                x1 = max(0, x - margin_x)
                y1 = max(0, y - margin_y)
                x2 = min(frame.shape[1], x + width + margin_x)
                y2 = min(frame.shape[0], y + height + margin_y)

                face_bgr = frame[y1:y2, x1:x2]
                if face_bgr.size == 0:
                    continue

                face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
                class_name, confidence, cosine = classifier.predict_rgb(face_rgb)[0]

                if class_name == "UNKNOWN":
                    text = f"UNKNOWN cos={cosine:.3f}"
                else:
                    text = (
                        f"{class_name} conf={confidence:.3f} cos={cosine:.3f}"
                    )

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    text,
                    (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("ArcFace camera recognition", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Test ArcFace/IResNet classification results."
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--checkpoint", required=True)
    common.add_argument("--device", default="auto")
    common.add_argument(
        "--unknown-threshold",
        type=float,
        default=0.0,
        help="Reject the best class when cosine similarity is below this value.",
    )

    folder_parser = subparsers.add_parser("folder", parents=[common])
    folder_parser.add_argument("--input-dir", required=True)
    folder_parser.add_argument("--recursive", action="store_true")
    folder_parser.add_argument("--top-k", type=int, default=1)
    folder_parser.add_argument("--output-csv", default="")
    folder_parser.set_defaults(function=run_folder_mode)

    camera_parser = subparsers.add_parser("camera", parents=[common])
    camera_parser.add_argument("--camera-id", type=int, default=0)
    camera_parser.add_argument("--width", type=int, default=1280)
    camera_parser.add_argument("--height", type=int, default=720)
    camera_parser.add_argument("--min-face-size", type=int, default=80)
    camera_parser.add_argument("--scale-factor", type=float, default=1.1)
    camera_parser.add_argument("--min-neighbors", type=int, default=5)
    camera_parser.add_argument("--face-margin", type=float, default=0.20)
    camera_parser.set_defaults(function=run_camera_mode)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
