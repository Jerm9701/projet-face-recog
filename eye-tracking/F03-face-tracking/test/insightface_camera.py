import argparse
import cv2
import time
import numpy as np
import insightface
from insightface.app import FaceAnalysis


def run_camera_detection(camera_index: int = 0, model_name: str = 'buffalo_l', provider: str = 'CUDAExecutionProvider', det_size: int = 640, device: int = 0, max_fps: float = 0.0, min_score: float = 0.8):
    """Run InsightFace detection on webcam input."""
    app = FaceAnalysis(name=model_name, allowed_modules=['detection'], providers=[provider])
    det_size_value = None if det_size <= 0 else (det_size, det_size)
    # use GPU context when CUDA provider requested
    ctx_id = -1
    if 'CUDA' in provider.upper():
        ctx_id = int(device)
    app.prepare(ctx_id=ctx_id, det_size=det_size_value, det_thresh=min_score)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {camera_index}")

    print("Starting camera face detection. Press 'q' to quit.")
    prev_time = time.time()
    fps = 0.0
    smooth_alpha = 0.05
    last_proc_time = 0.0
    target_dt = 0.0 if max_fps <= 0 else 1.0 / float(max_fps)
    no_detect_count = 0
    last_face_crop = np.zeros((160, 160, 3), dtype=np.uint8)
    cv2.putText(last_face_crop, '[NO DETECTION]', (2, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow('Top Face', last_face_crop)
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Failed to read frame from camera. Exiting.")
            break

        # update FPS
        now = time.time()
        dt = now - prev_time if now > prev_time else 1e-6
        prev_time = now
        current_fps = 1.0 / dt if dt > 0 else 0.0
        fps = (1.0 - smooth_alpha) * fps + smooth_alpha * current_fps if fps > 0 else current_fps

        # Throttle processing if max_fps set: skip detection when called too quickly
        do_detect = True
        if target_dt > 0.0:
            if (now - last_proc_time) < target_dt:
                do_detect = False

        faces = []
        if do_detect:
            faces = app.get(frame)
            last_proc_time = now
            if faces:
                frame = app.draw_on(frame, faces)

        # draw FPS on frame
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        # display highest-confidence face in its own window
        if faces:
            no_detect_count = 0
            top_face = max(faces, key=lambda f: f.det_score)
            x1, y1, x2, y2 = top_face.bbox.astype(int)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)
            if x2 > x1 and y2 > y1:
                face_crop = frame[y1:y2, x1:x2].copy()
                #last_face_crop = cv2.resize(face_crop, (160, 160), interpolation=cv2.INTER_LINEAR)
                cv2.imshow('Top Face', face_crop)
        else:
            no_detect_count += 1
            no_face = face_crop.copy()
            if no_detect_count > 10:
                cv2.putText(no_face, '[WAITING FOR DETECTION]', (2, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow('Top Face', no_face)

        cv2.imshow('InsightFace Camera Detection', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='InsightFace camera face detection')
    parser.add_argument('--camera', type=int, default=0, help='Camera index (default: 0)')
    parser.add_argument('--model', type=str, default='buffalo_l', help='InsightFace model name (default: buffalo_l)')
    parser.add_argument('--provider', type=str, default='CUDAExecutionProvider', help='ONNX provider to use (default: CUDAExecutionProvider)')
    parser.add_argument('--device', type=int, default=0, help='CUDA device id (default: 0)')
    parser.add_argument('--max-fps', type=float, default=0.0, help='Maximum detection FPS (0 = unlimited)')
    parser.add_argument('--min-score', type=float, default=0.8, help='Minimum detection score threshold (default: 0.8)')
    parser.add_argument('--det-size', type=int, default=640, help='Detection size (default: 640). Set 0 for auto.')
    args = parser.parse_args()

    run_camera_detection(camera_index=args.camera, model_name=args.model, provider=args.provider, det_size=args.det_size, device=args.device, max_fps=args.max_fps, min_score=args.min_score)
