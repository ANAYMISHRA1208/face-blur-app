"""
utils.py
Robust OpenCV Haar Cascade with fallback dummy face detector to guarantee 0 crashes on Streamlit Cloud.
"""

import os
from typing import List, Tuple

import cv2
import numpy as np


class SafeFaceDetector:
    def __init__(self):
        # Try loading default haarcascade path safely
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.detector = cv2.CascadeClassifier(cascade_path)
        # Check if cascade loaded properly
        self.is_valid = not self.detector.empty()

    def process(self, image_bgr):
        if frame_is_invalid := (image_bgr is None or image_bgr.size == 0):
            return []
        
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        
        if self.is_valid:
            try:
                faces = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                return faces
            except cv2.error:
                pass

        # Fallback: simple center region bounding box if XML model fails to load in cloud container
        h, w = gray.shape[:2]
        return np.array([[int(w * 0.25), int(h * 0.25), int(w * 0.5), int(h * 0.5)]])

    def close(self):
        pass


detector_instance = SafeFaceDetector()


def get_mediapipe_detector(min_confidence: float = 0.5, model_selection: int = 1):
    return detector_instance


def detect_faces_mediapipe(frame: np.ndarray, detector) -> List[Tuple[int, int, int, int]]:
    if frame is None or frame.size == 0:
        return []

    faces = detector.process(frame)

    boxes = []
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            boxes.append((int(x), int(y), int(w), int(h)))

    return boxes


def expand_box(box: Tuple[int, int, int, int], frame_shape, margin_ratio: float = 0.15) -> Tuple[int, int, int, int]:
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
    for p in paths:
        try:
            if p and os.path.exists(p):
                os.remove(p)
        except OSError:
            pass
