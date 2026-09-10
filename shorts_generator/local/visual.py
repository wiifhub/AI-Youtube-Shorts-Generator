"""Lightweight local visual-event analysis for highlight ranking."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List


def analyze_video(media_path: str, sample_seconds: float = 1.0) -> List[Dict]:
    """Find scene-change and face/reaction signals without a second ML model."""
    try:
        import cv2  # type: ignore
    except ImportError:
        return []

    cap = cv2.VideoCapture(str(Path(media_path)))
    if not cap.isOpened():
        return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    duration = frame_count / fps if frame_count else 0.0
    step = max(1, int(fps * max(0.25, sample_seconds)))
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    previous = None
    events: List[Dict] = []
    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % step:
            frame_index += 1
            continue
        timestamp = frame_index / fps
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (160, 90))
        face_count = len(cascade.detectMultiScale(gray, 1.1, 5, minSize=(35, 35)))
        if previous is not None:
            change = float(cv2.absdiff(small, previous).mean())
            if change >= 24.0:
                events.append({"time": round(timestamp, 2), "type": "scene_change", "score": round(min(1.0, change / 80.0), 3)})
        if face_count >= 2:
            events.append({"time": round(timestamp, 2), "type": "multiple_speakers", "score": min(1.0, face_count / 4.0)})
        elif face_count == 1:
            events.append({"time": round(timestamp, 2), "type": "speaker_visible", "score": 0.35})
        previous = small
        frame_index += 1
        if duration and timestamp >= duration:
            break
    cap.release()
    return events


def extract_thumbnail(media_path: str, timestamp: float, out_path: str) -> str:
    """Extract a high-quality JPG frame for a short's thumbnail."""
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError("opencv-python is required for thumbnail extraction") from exc
    cap = cv2.VideoCapture(str(Path(media_path)))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {media_path}")
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(timestamp)) * 1000.0)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"could not read thumbnail frame at {timestamp:.1f}s")
    if not cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92]):
        raise RuntimeError(f"could not write thumbnail {out_path}")
    return out_path
