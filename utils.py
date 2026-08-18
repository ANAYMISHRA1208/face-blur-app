"""
utils.py
Helper functions for face detection, blurring, and video/audio processing.
"""

import os
from typing import List, Tuple

import cv2
import numpy as np
import mediapipe as mp


class MediaPipeFaceDetectorWrapper:
    def __init__(self, min_confidence=0.5, model_selection=1):
        # MediaPipe solution fallback setup
        try:
            self.mp_face = mp.solutions.face_detection
        except AttributeError:
            from mediapipe.python.solutions import face_detection as mp_face
            self.mp_face = mp_face
            
        self.detector = self.mp_face.FaceDetection(
            min_detection_confidence=min_confidence,
            model_selection=model_selection
        )

    def process(self, image):
        return self.detector.process(image)

    def close(self):
        if hasattr(self.detector, 'close'):
            self.detector.close()


def get_mediapipe_detector(min_confidence: float = 0.5, model_selection: int = 1):
    return MediaPipeFaceDetectorWrapper(min_confidence, model_selection)


def detect_faces_mediapipe(frame: np.ndarray, detector) -> List[Tuple[int, int, int, int]]:
    """
    Detect faces in a BGR frame using MediaPipe.
    Returns a list of bounding boxes as (x, y, w, h) in pixel coordinates.
    """
    if frame is None or frame.size == 0:
        return []

    h, w = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.process(rgb_frame)

    boxes = []
    if results and results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            box_w = int(bbox.width * w)
            box_h = int(bbox.height * h)

            x = max(0, x)
            y = max(0, y)
            box_w = min(box_w, w - x)
            box_h = min(box_h, h - y)

            if box_w > 0 and box_h > 0:
                boxes.append((x, y, box_w, box_h))

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
