"""
tracker.py
Face identity extraction, deduplication, and selective tracking/blurring
across a video using `face_recognition` embeddings (accurate but slower)
combined with MediaPipe detection (fast) for a good speed/accuracy balance.

Two-pass design for Mode 2 (Selective Blur):
  Pass 1 (extract_unique_faces): sparsely sample frames, detect + encode
      faces, and deduplicate them into a FaceDatabase of unique identities.
  Pass 2 (process_video_selective): walk every frame, detect faces every
      frame (cheap), but only re-run the expensive encoding+matching every
      `encode_every_n_frames` frames. In between, identities are carried
      over via simple nearest-centroid tracking.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import cv2
import numpy as np
import face_recognition

from utils import blur_region, expand_box, get_mediapipe_detector, detect_faces_mediapipe, pixelate_region


@dataclass
class UniqueFace:
    """Represents one distinct identity discovered during the scan pass."""
    face_id: int
    encoding: np.ndarray
    thumbnail: np.ndarray  # small BGR crop, for UI display
    sample_count: int = 1


class FaceDatabase:
    """
    Holds unique face encodings discovered during the scan pass. A new
    encoding is compared against existing ones; a match within `tolerance`
    is treated as the same person and its encoding is running-averaged.
    """

    def __init__(self, tolerance: float = 0.5):
        self.tolerance = tolerance
        self.faces: List[UniqueFace] = []
        self._next_id = 1

    def match_or_add(self, encoding: np.ndarray, thumbnail: np.ndarray) -> int:
        if self.faces:
            known_encodings = [f.encoding for f in self.faces]
            distances = face_recognition.face_distance(known_encodings, encoding)
            best_idx = int(np.argmin(distances))
            if distances[best_idx] <= self.tolerance:
                matched = self.faces[best_idx]
                matched.encoding = (matched.encoding * matched.sample_count + encoding) / (matched.sample_count + 1)
                matched.sample_count += 1
                return matched.face_id

        new_face = UniqueFace(face_id=self._next_id, encoding=encoding, thumbnail=thumbnail)
        self.faces.append(new_face)
        self._next_id += 1
        return new_face.face_id

    def match_only(self, encoding: np.ndarray) -> Optional[int]:
        """Match against the existing database without adding new entries (pass 2)."""
        if not self.faces:
            return None
        known_encodings = [f.encoding for f in self.faces]
        distances = face_recognition.face_distance(known_encodings, encoding)
        best_idx = int(np.argmin(distances))
        if distances[best_idx] <= self.tolerance:
            return self.faces[best_idx].face_id
        return None

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
    Pass 1: Scan the video, sampling every N frames, detect faces with
    MediaPipe, compute face_recognition encodings, and build a deduplicated
    FaceDatabase of unique identities with representative thumbnails.
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
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # face_recognition expects (top, right, bottom, left)
                    fr_boxes = [(y, x + w, y + h, x) for (x, y, w, h) in boxes]
                    encodings = face_recognition.face_encodings(
                        rgb_frame, known_face_locations=fr_boxes, num_jitters=1
                    )

                    for box, encoding in zip(boxes, encodings):
                        ex, ey, ew, eh = expand_box(box, frame.shape, margin_ratio=0.2)
                        thumb = frame[ey:ey + eh, ex:ex + ew].copy()
                        if thumb.size > 0:
                            thumb = cv2.resize(thumb, (120, 120))
                            db.match_or_add(encoding, thumb)

                scanned += 1
                if progress_callback and total_frames > 0:
                    progress_callback(min(frame_idx / total_frames, 1.0))

                if scanned >= max_frames_to_scan:
                    break

            frame_idx += 1
    finally:
        cap.release()
        detector.close()

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
    Pass 2: Re-process the full video frame-by-frame.
      - Every frame: detect face boxes with MediaPipe (cheap).
      - Every `encode_every_n_frames` frames: compute face_recognition
        encodings and re-match identities against the database.
      - In between: reuse the last known identity via nearest-centroid
        tracking, which is far cheaper than encoding on every single frame.
    Only faces whose matched identity is in `selected_ids` get blurred.
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

    tracked_faces: List[Dict] = []  # [{"centroid": (cx, cy), "face_id": int|None, "box": (...)}]
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            boxes = detect_faces_mediapipe(frame, detector)
            do_full_match = (frame_idx % encode_every_n_frames == 0) and boxes

            new_tracked = []

            if do_full_match:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                fr_boxes = [(y, x + w, y + h, x) for (x, y, w, h) in boxes]
                encodings = face_recognition.face_encodings(
                    rgb_frame, known_face_locations=fr_boxes, num_jitters=0
                )
                for box, encoding in zip(boxes, encodings):
                    face_id = db.match_only(encoding)
                    x, y, w, h = box
                    centroid = (x + w / 2, y + h / 2)
                    new_tracked.append({"centroid": centroid, "face_id": face_id, "box": box})
            else:
                # Cheap path: match current detections to the previous frame's
                # tracked identities by nearest centroid — no re-encoding needed.
                for box in boxes:
                    x, y, w, h = box
                    centroid = (x + w / 2, y + h / 2)
                    best_match = None
                    best_dist = float("inf")
                    for t in tracked_faces:
                        d = (t["centroid"][0] - centroid[0]) ** 2 + (t["centroid"][1] - centroid[1]) ** 2
                        if d < best_dist:
                            best_dist = d
                            best_match = t
                    # Only carry over an identity if the closest previous face
                    # is reasonably near, to avoid mislabeling a new/different face.
                    face_id = best_match["face_id"] if best_match and best_dist < (width * 0.15) ** 2 else None
                    new_tracked.append({"centroid": centroid, "face_id": face_id, "box": box})

            for t in new_tracked:
                if t["face_id"] is not None and t["face_id"] in selected_ids:
                    ex, ey, ew, eh = expand_box(t["box"], frame.shape, margin_ratio=0.2)
                    frame = blur_region(frame, (ex, ey, ew, eh), intensity=blur_intensity)

            tracked_faces = new_tracked
            writer.write(frame)

            if progress_callback and total_frames > 0:
                progress_callback(min(frame_idx / total_frames, 1.0))

            frame_idx += 1
    finally:
        cap.release()
        writer.release()
        detector.close()

    return output_path


def process_video_auto_blur(
    video_path: str,
    output_path: str,
    blur_intensity: str = "high",
    use_pixelate: bool = False,
    progress_callback=None,
) -> str:
    """
    Mode 1 (video): detect every face in every frame with MediaPipe and blur
    it. No identity tracking is needed since ALL faces get blurred.
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
        detector.close()

    return output_path
