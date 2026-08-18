"""
tracker.py
Lightweight face detection and selective blurring using OpenCV.
Features clean individual cropping and basic visual deduplication.
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
    Maintains detected unique faces with visual thumbnail comparison.
    """

    def __init__(self, tolerance: float = 0.5):
        self.tolerance = tolerance
        self.faces: List[UniqueFace] = []
        self._next_id = 1

    def is_duplicate(self, new_thumb: np.ndarray) -> bool:
        """Check if a visually similar face thumbnail already exists."""
        if not self.faces:
            return False
        
        new_gray = cv2.cvtColor(new_thumb, cv2.COLOR_BGR2GRAY)
        
        for face in self.faces:
            old_gray = cv2.cvtColor(face.thumbnail, cv2.COLOR_BGR2GRAY)
            # Match template or mean absolute difference
            diff = cv2.absdiff(new_gray, old_gray)
            mean_diff = np.mean(diff)
            if mean_diff < 40:  # Similarity threshold
                return True
        return False

    def match_or_add(self, thumbnail: np.ndarray) -> int:
        if not self.is_duplicate(thumbnail):
            new_face = UniqueFace(face_id=self._next_id, thumbnail=thumbnail)
            self.faces.append(new_face)
            self._next_id += 1
            return new_face.face_id
        return -1

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
    Pass 1: Scan video and extract clean individual face thumbnails.
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
                        x, y, w, h = box
                        # Direct precise face crop without extra margin expansion
                        thumb = frame[y:y + h, x:x + w].copy()
                        if thumb.size > 0 and w > 20 and h > 20:
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
    Pass 2: Process video and blur detected faces.
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
                ex, ey, ew, eh = expand_box(box, frame.shape, margin_ratio=0.15)
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
    Mode 1 (video): Detect every face in every frame and blur it.
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
                ex, ey, ew, eh = expand_box(box, frame.shape, margin_ratio=0.15)
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
