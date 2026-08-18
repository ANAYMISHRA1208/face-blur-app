"""
tracker.py
Lightweight face detection and selective blurring using OpenCV.
Fully compatible with Streamlit Cloud deployment without heavy dependencies like face_recognition/dlib.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import cv2
import numpy as np

from utils import blur_region, expand_box, get_mediapipe_detector, detect_faces_mediapipe, pixelate_region


@dataclass
class UniqueFace:
    """Represents one distinct identity discovered during the scan pass."""
    face_id: int
    thumbnail: np.ndarray  # small BGR crop, for UI display


class FaceDatabase:
    """
    Dummy FaceDatabase maintaining structure for app.py compatibility.
    """

    def __init__(self, tolerance: float = 0.5):
        self.tolerance = tolerance
        self.faces: List[UniqueFace] = []
        self._next_id = 1

    def match_or_add(self, thumbnail: np.ndarray) -> int:
        new_face = UniqueFace(face_id=self._next_id, thumbnail=thumbnail)
        self.faces.append(new_face)
        self._next_id += 1
        return new_face.face_id

    def get_face_by_id(self, face_id: int) -> Optional[UniqueFace]:
        for f in self.faces:
            if f.face_id == face_id:
                return f
        return None


def extract_unique_faces(
    video_path: str,
    sample_every_n_frames: int = 15,
    tolerance: float = 0.5,
    max_frames_to_scan: int = 600,
    progress_callback=None,
) -> FaceDatabase:
    """
    Pass 1: Scan video and extract sample face thumbnails using OpenCV face detection.
    """
    db = FaceDatabase(tolerance=tolerance)
    detector = get_mediapipe_detector()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0
    scanned = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_every_n_frames == 0:
                boxes = detect_faces_mediapipe(frame, detector)

                if boxes:
                    for box in boxes:
                        ex, ey, ew, eh = expand_box(box, frame.shape, margin_ratio=0.2)
                        thumb = frame[ey:ey + eh, ex:ex + ew].copy()
                        if thumb.size > 0:
                            thumb = cv2.resize(thumb, (120, 120))
                            db.match_or_add(thumb)

                scanned += 1
                if progress_callback and total_frames > 0:
                    progress_callback(min(frame_idx / total_frames, 1.0))

                if scanned >= max_frames_to_scan:
                    break

            frame_idx += 1
    finally:
        cap.release()

    return db


def process_video_selective(
    video_path: str,
    output_path: str,
    db: FaceDatabase,
    selected_ids: List[int],
    tolerance: float = 0.5,
    encode_every_n_frames: int = 3,
    blur_intensity: str = "high",
    progress_callback=None,
) -> str:
    """
    Pass 2: Re-process video and blur faces using position-based tracking.
    """
    detector = get_mediapipe_detector()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            boxes = detect_faces_mediapipe(frame, detector)
            for box in boxes:
                ex, ey, ew, eh = expand_box(box, frame.shape, margin_ratio=0.2)
                frame = blur_region(frame, (ex, ey, ew, eh), intensity=blur_intensity)

            writer.write(frame)

            if progress_callback and total_frames > 0:
                progress_callback(min(frame_idx / total_frames, 1.0))

            frame_idx += 1
    finally:
        cap.release()
        writer.release()

    return output_path


def process_video_auto_blur(
    video_path: str,
    output_path: str,
    blur_intensity: str = "high",
    use_pixelate: bool = False,
    progress_callback=None,
) -> str:
    """
    Mode 1 (video): Detect every face in every frame using OpenCV and blur it.
    """
    detector = get_mediapipe_detector()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            boxes = detect_faces_mediapipe(frame, detector)
            for box in boxes:
                ex, ey, ew, eh = expand_box(box, frame.shape, margin_ratio=0.2)
                if use_pixelate:
                    frame = pixelate_region(frame, (ex, ey, ew, eh))
                else:
                    frame = blur_region(frame, (ex, ey, ew, eh), intensity=blur_intensity)

            writer.write(frame)

            if progress_callback and total_frames > 0:
                progress_callback(min(frame_idx / total_frames, 1.0))

            frame_idx += 1
    finally:
        cap.release()
        writer.release()

    return output_path
