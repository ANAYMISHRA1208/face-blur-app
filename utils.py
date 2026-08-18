"""
utils.py
Helper functions for face detection, blurring, and video/audio processing using OpenCV.
"""

import os
from typing import List, Tuple

import cv2
import numpy as np

# Load OpenCV built-in Haar Cascade Face Detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def get_mediapipe_detector(min_confidence: float = 0.5, model_selection: int = 1):
    """
    Dummy wrapper to keep compatibility with app.py & tracker.py
    """
    return face_cascade


def detect_faces_mediapipe(frame: np.ndarray, detector) -> List[Tuple[int, int, int, int]]:
    """
    Detect faces in a BGR frame using OpenCV Haar Cascade.
    Returns a list of bounding boxes as (x, y, w, h) in pixel coordinates.
    """
    if frame is None or frame.size == 0:
        return []

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    boxes = []
    for (x, y, w, h) in faces:
        boxes.append((int(x), int(y), int(w), int(h)))

    return boxes


def expand_box(box: Tuple[int, int, int, int], frame_shape, margin_ratio: float = 0.15) -> Tuple[int, int, int, int]:
    """
    Slightly enlarge a bounding box so the blur fully covers hair/chin/ears.
    """
    x, y, w, h = box
    frame_h, frame_w = frame_shape[:2]

    mx = int(w * margin_ratio)
    my = int(h * margin_ratio)

    x1 = max(0, x - mx)
    y1 = max(0, y - my)
    x2 = min(frame_w, x + w + mx)
    y2 = min(frame_h, y + h + my)

    return x1, y1, x2 - x1, y2 - y1


def blur_region(frame: np.ndarray, box: Tuple[int, int, int, int], intensity: str = "high") -> np.ndarray:
    """
    Apply Gaussian blur to a rectangular region of a frame.
    """
    x, y, w, h = box
    if w <= 0 or h <= 0:
        return frame

    roi = frame[y:y + h, x:x + w]
    if roi.size == 0:
        return frame

    kernel_map = {"low": 15, "medium": 35, "high": 55}
    k = kernel_map.get(intensity, 45)

    k = max(k, (min(w, h) // 2) | 1)
    if k % 2 == 0:
        k += 1

    blurred_roi = cv2.GaussianBlur(roi, (k, k), 0)
    frame[y:y + h, x:x + w] = blurred_roi
    return frame


def pixelate_region(frame: np.ndarray, box: Tuple[int, int, int, int], blocks: int = 10) -> np.ndarray:
    """
    Alternative anonymization style — mosaic/pixelate instead of Gaussian blur.
    """
    x, y, w, h = box
    if w <= 0 or h <= 0:
        return frame

    roi = frame[y:y + h, x:x + w]
    if roi.size == 0:
        return frame

    small = cv2.resize(roi, (blocks, blocks), interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    frame[y:y + h, x:x + w] = pixelated
    return frame


def merge_audio_with_video(original_video_path: str, processed_video_no_audio_path: str, final_output_path: str) -> bool:
    """
    Attach the original video's audio track onto the newly processed video.
    """
    try:
        from moviepy.editor import VideoFileClip
    except ImportError:
        from moviepy.video.io.VideoFileClip import VideoFileClip

    original_clip = None
    processed_clip = None
    try:
        original_clip = VideoFileClip(original_video_path)
        processed_clip = VideoFileClip(processed_video_no_audio_path)

        if original_clip.audio is not None:
            final_clip = processed_clip.set_audio(original_clip.audio)
            final_clip.write_videofile(
                final_output_path,
                codec="libx264",
                audio_codec="aac",
                verbose=False,
                logger=None,
            )
            final_clip.close()
            return True
        else:
            processed_clip.write_videofile(
                final_output_path,
                codec="libx264",
                verbose=False,
                logger=None,
            )
            return False
    finally:
        if original_clip is not None:
            original_clip.close()
        if processed_clip is not None:
            processed_clip.close()


def cleanup_temp_files(*paths: str) -> None:
    """Safely delete temporary files, ignoring missing files or lock errors."""
    for p in paths:
        try:
            if p and os.path.exists(p):
                os.remove(p)
        except OSError:
            pass
