#python .\infer_retinaface_r50.py camera --checkpoint .\runs\retinaface_r50\best.pt --score-threshold 0.7
#python .\infer_retinaface_r50.py image --checkpoint .\runs\retinaface_r50\best.pt --input-image ..\..\captures\rd_img1.jpg
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torchvision.models import resnet50
from torchvision.ops import nms


class ResNet50Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        network = resnet50(weights=None)
        self.stem = nn.Sequential(network.conv1, network.bn1, network.relu, network.maxpool)
        self.layer1 = network.layer1
        self.layer2 = network.layer2
        self.layer3 = network.layer3
        self.layer4 = network.layer4

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        x = self.stem(x)
        x = self.layer1(x)
        c3 = self.layer2(x)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return c3, c4, c5


class FPN(nn.Module):
    def __init__(self, out_channels: int = 128) -> None:
        super().__init__()
        self.l3 = nn.Conv2d(512, out_channels, 1)
        self.l4 = nn.Conv2d(1024, out_channels, 1)
        self.l5 = nn.Conv2d(2048, out_channels, 1)
        self.o3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.o4 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.o5 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, c3: Tensor, c4: Tensor, c5: Tensor) -> List[Tensor]:
        p5 = self.l5(c5)
        p4 = self.l4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p3 = self.l3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        return [self.o3(p3), self.o4(p4), self.o5(p5)]


class DetectionHead(nn.Module):
    def __init__(self, channels: int, anchors_per_location: int = 2) -> None:
        super().__init__()
        self.tower = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.cls = nn.Conv2d(channels, anchors_per_location, 1)
        self.box = nn.Conv2d(channels, anchors_per_location * 4, 1)
        self.lmk = nn.Conv2d(channels, anchors_per_location * 10, 1)

    def forward(self, feature: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        x = self.tower(feature)
        return self.cls(x), self.box(x), self.lmk(x)


class RetinaFaceLite(nn.Module):
    def __init__(self, fpn_channels: int = 128) -> None:
        super().__init__()
        self.backbone = ResNet50Backbone()
        self.fpn = FPN(fpn_channels)
        self.head = DetectionHead(fpn_channels, 2)

    def forward(self, images: Tensor) -> Dict[str, Tensor]:
        c3, c4, c5 = self.backbone(images)
        features = self.fpn(c3, c4, c5)
        cls, boxes, landmarks = [], [], []
        for feature in features:
            c, b, l = self.head(feature)
            batch = feature.shape[0]
            cls.append(c.permute(0, 2, 3, 1).reshape(batch, -1))
            boxes.append(b.permute(0, 2, 3, 1).reshape(batch, -1, 4))
            landmarks.append(l.permute(0, 2, 3, 1).reshape(batch, -1, 10))
        return {
            "classification": torch.cat(cls, dim=1),
            "boxes": torch.cat(boxes, dim=1),
            "landmarks": torch.cat(landmarks, dim=1),
        }


def generate_anchors(image_size: int, strides: Sequence[int], anchor_sizes: Sequence[Sequence[int]], device: torch.device) -> Tensor:
    anchors = []
    for stride, sizes in zip(strides, anchor_sizes):
        feature_size = math.ceil(image_size / stride)
        for y in range(feature_size):
            cy = (y + 0.5) * stride
            for x in range(feature_size):
                cx = (x + 0.5) * stride
                for size in sizes:
                    h = size / 2.0
                    anchors.append([cx - h, cy - h, cx + h, cy + h])
    return torch.tensor(anchors, dtype=torch.float32, device=device)


def decode_boxes(anchors: Tensor, deltas: Tensor) -> Tensor:
    aw = anchors[:, 2] - anchors[:, 0]
    ah = anchors[:, 3] - anchors[:, 1]
    acx = (anchors[:, 0] + anchors[:, 2]) * 0.5
    acy = (anchors[:, 1] + anchors[:, 3]) * 0.5
    cx = deltas[:, 0] * aw + acx
    cy = deltas[:, 1] * ah + acy
    w = torch.exp(deltas[:, 2].clamp(max=10.0)) * aw
    h = torch.exp(deltas[:, 3].clamp(max=10.0)) * ah
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=1)


def decode_landmarks(anchors: Tensor, deltas: Tensor) -> Tensor:
    aw = anchors[:, 2] - anchors[:, 0]
    ah = anchors[:, 3] - anchors[:, 1]
    acx = (anchors[:, 0] + anchors[:, 2]) * 0.5
    acy = (anchors[:, 1] + anchors[:, 3]) * 0.5
    deltas = deltas.reshape(-1, 5, 2)
    landmarks = torch.empty_like(deltas)
    landmarks[:, :, 0] = deltas[:, :, 0] * aw[:, None] + acx[:, None]
    landmarks[:, :, 1] = deltas[:, :, 1] * ah[:, None] + acy[:, None]
    return landmarks


class FaceDetector:
    def __init__(
        self,
        checkpoint_path: str,
        device_name: str = "auto",
        use_cpu: bool = False,
        score_threshold: float | None = None,
        nms_threshold: float | None = None,
        max_detections: int | None = None,
    ) -> None:
        if use_cpu:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(
                "cuda" if device_name == "auto" and torch.cuda.is_available() else ("cpu" if device_name == "auto" else device_name)
            )
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        config = checkpoint.get("config", {})
        self.image_size = int(config.get("image_size", 512))
        self.fpn_channels = int(config.get("fpn_channels", 128))
        self.strides = tuple(int(v) for v in config.get("strides", [8, 16, 32]))
        self.anchor_sizes = tuple(tuple(int(s) for s in level) for level in config.get("anchor_sizes", [[16, 32], [64, 128], [256, 512]]))
        self.score_threshold = float(score_threshold if score_threshold is not None else config.get("score_threshold", 0.35))
        self.nms_threshold = float(nms_threshold if nms_threshold is not None else config.get("nms_threshold", 0.4))
        self.max_detections = int(max_detections if max_detections is not None else config.get("max_detections", 750))
        self.model = RetinaFaceLite(self.fpn_channels)
        self.model.load_state_dict(checkpoint["model"])
        self.model.to(self.device).eval()
        self.anchors = generate_anchors(self.image_size, self.strides, self.anchor_sizes, self.device)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        print(f"Loaded detector on {self.device}; threshold={self.score_threshold:.2f}")

    def preprocess(self, image: np.ndarray) -> Tuple[Tensor, float, float]:
        h, w = image.shape[:2]
        frame_tensor = torch.from_numpy(np.ascontiguousarray(image)).to(self.device, non_blocking=True)
        frame_tensor = frame_tensor.permute(2, 0, 1)
        frame_tensor = frame_tensor[[2, 1, 0]]  # BGR -> RGB
        frame_tensor = frame_tensor.unsqueeze(0).float() / 255.0
        frame_tensor = F.interpolate(
            frame_tensor,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        frame_tensor = (frame_tensor - self.mean) / self.std
        return frame_tensor, w / self.image_size, h / self.image_size

    @torch.inference_mode()
    def detect(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        tensor, sx, sy = self.preprocess(image)
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"):
            pred = self.model(tensor)
        scores = torch.sigmoid(pred["classification"][0])
        mask = scores >= self.score_threshold
        if not mask.any():
            return {"boxes": np.zeros((0, 4), np.float32), "scores": np.zeros((0,), np.float32), "landmarks": np.zeros((0, 5, 2), np.float32)}
        scores = scores[mask]
        anchors = self.anchors[mask]
        boxes = decode_boxes(anchors, pred["boxes"][0][mask])
        landmarks = decode_landmarks(anchors, pred["landmarks"][0][mask])
        if scores.numel() > 5000:
            scores, idx = scores.topk(5000)
            boxes = boxes[idx]
            landmarks = landmarks[idx]
        keep = nms(boxes, scores, self.nms_threshold)[:self.max_detections]
        boxes = boxes[keep]
        scores = scores[keep]
        landmarks = landmarks[keep]
        boxes[:, [0, 2]] *= sx
        boxes[:, [1, 3]] *= sy
        landmarks[:, :, 0] *= sx
        landmarks[:, :, 1] *= sy
        return {"boxes": boxes.float().cpu().numpy(), "scores": scores.float().cpu().numpy(), "landmarks": landmarks.float().cpu().numpy()}


LANDMARK_COLORS = [(255, 0, 0), (0, 255, 255), (0, 0, 255), (255, 0, 255), (0, 165, 255)]


def draw_detections(image: np.ndarray, detections: Dict[str, np.ndarray]) -> np.ndarray:
    out = image.copy()
    for box, score, landmarks in zip(detections["boxes"], detections["scores"], detections["landmarks"]):
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, f"face {score:.3f}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
        for point, color in zip(landmarks, LANDMARK_COLORS):
            px, py = point.astype(int)
            cv2.circle(out, (px, py), 3, color, -1, cv2.LINE_AA)
    return out


def create_status_frame(width: int, height: int, message: str) -> np.ndarray:
    width, height = max(width, 640), max(height, 360)
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2
    (tw, th), _ = cv2.getTextSize(message, font, scale, thickness)
    cv2.putText(frame, message, ((width - tw) // 2, (height + th) // 2), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return frame


def add_waiting_message(frame: np.ndarray) -> np.ndarray:
    out = frame.copy()
    message = "[waiting for detection]"
    cv2.rectangle(out, (10, out.shape[0] - 60), (390, out.shape[0] - 10), (0, 0, 0), -1)
    cv2.putText(out, message, (20, out.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    return out


def run_image(args: argparse.Namespace) -> None:
    detector = FaceDetector(
        args.checkpoint,
        device_name=args.device,
        use_cpu=args.cpu,
        score_threshold=args.score_threshold,
        nms_threshold=args.nms_threshold,
        max_detections=args.max_detections,
    )
    image = cv2.imread(str(Path(args.input_image)))
    if image is None:
        raise FileNotFoundError(args.input_image)
    detections = detector.detect(image)
    display = draw_detections(image, detections) if len(detections["boxes"]) else create_status_frame(image.shape[1], image.shape[0], "[no face detected]")
    cv2.imshow(args.window_name, display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_camera(args: argparse.Namespace) -> None:
    detector = FaceDetector(
        args.checkpoint,
        device_name=args.device,
        use_cpu=args.cpu,
        score_threshold=args.score_threshold,
        nms_threshold=args.nms_threshold,
        max_detections=args.max_detections,
    )
    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {args.camera_id}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
    first_detection_seen = False
    missing_count = 0
    last_detected_frame = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            detections = detector.detect(frame)
            if len(detections["boxes"]) > 0:
                display = draw_detections(frame, detections)
                last_detected_frame = display.copy()
                first_detection_seen = True
                missing_count = 0
            else:
                missing_count += 1
                if not first_detection_seen:
                    display = create_status_frame(frame.shape[1], frame.shape[0], "[no face detected]")
                else:
                    display = last_detected_frame.copy()
                    if missing_count >= 5:
                        display = add_waiting_message(display)
            cv2.imshow(args.window_name, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--checkpoint", required=True)
    common.add_argument("--device", default="auto")
    common.add_argument("--cpu", action="store_true", help="Force execution on CPU")
    common.add_argument("--score-threshold", type=float, default=None)
    common.add_argument("--nms-threshold", type=float, default=None)
    common.add_argument("--max-detections", type=int, default=None)
    common.add_argument("--window-name", default="RetinaFace-like detector")
    p1 = sub.add_parser("image", parents=[common])
    p1.add_argument("--input-image", required=True)
    p1.set_defaults(func=run_image)
    p2 = sub.add_parser("camera", parents=[common])
    p2.add_argument("--camera-id", type=int, default=0)
    p2.add_argument("--camera-width", type=int, default=1280)
    p2.add_argument("--camera-height", type=int, default=720)
    p2.set_defaults(func=run_camera)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
