"""
app.py
Streamlit dashboard for the Face Blurring Application.

Mode 1 — Automatic Blur: blurs every detected face in an uploaded photo or video.
Mode 2 — Selective Blur: scans a video for unique faces, lets the user pick
         which ones to blur, then re-renders the full video accordingly.

Run with:  streamlit run app.py
"""

import os
import tempfile

import cv2
import numpy as np
import streamlit as st

from utils import (
    get_mediapipe_detector,
    detect_faces_mediapipe,
    expand_box,
    blur_region,
    pixelate_region,
    merge_audio_with_video,
    cleanup_temp_files,
)
from tracker import extract_unique_faces, process_video_selective, process_video_auto_blur

st.set_page_config(page_title="Face Blur Studio", page_icon="🫥", layout="wide")

TEMP_DIR = tempfile.gettempdir()


def save_uploaded_file(uploaded_file) -> str:
    """Persist a Streamlit UploadedFile to a temp path and return that path."""
    suffix = os.path.splitext(uploaded_file.name)[1]
    tmp_path = os.path.join(TEMP_DIR, f"upload_{next(tempfile._get_candidate_names())}{suffix}")
    with open(tmp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return tmp_path


def blur_photo(image_path: str, intensity: str, use_pixelate: bool):
    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError("Could not read the uploaded image. Please try a different file.")

    detector = get_mediapipe_detector()
    try:
        boxes = detect_faces_mediapipe(frame, detector)
        for box in boxes:
            ex, ey, ew, eh = expand_box(box, frame.shape, margin_ratio=0.2)
            if use_pixelate:
                frame = pixelate_region(frame, (ex, ey, ew, eh))
            else:
                frame = blur_region(frame, (ex, ey, ew, eh), intensity=intensity)
    finally:
        detector.close()

    return frame, len(boxes)


def main():
    st.title("🫥 Face Blur Studio")
    st.caption("Automatic or selective face anonymization for photos and videos.")

    mode = st.sidebar.radio(
        "Choose a mode",
        ["Mode 1 — Automatic Blur", "Mode 2 — Selective Blur (Video only)"],
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Blur Settings")
    blur_intensity = st.sidebar.select_slider(
        "Blur intensity", options=["low", "medium", "high"], value="high"
    )
    use_pixelate = st.sidebar.checkbox("Use pixelation instead of Gaussian blur", value=False)

    if mode == "Mode 1 — Automatic Blur":
        run_mode_automatic(blur_intensity, use_pixelate)
    else:
        run_mode_selective(blur_intensity, use_pixelate)


def run_mode_automatic(blur_intensity, use_pixelate):
    st.header("Mode 1 — Automatic Blur")
    st.write("Upload a photo or video. Every detected face will be blurred automatically.")

    file_type = st.radio("File type", ["Photo", "Video"], horizontal=True)

    if file_type == "Photo":
        uploaded = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png", "bmp"])
        if uploaded and st.button("Blur Faces in Photo"):
            input_path = save_uploaded_file(uploaded)
            try:
                with st.spinner("Detecting and blurring faces..."):
                    result_frame, num_faces = blur_photo(input_path, blur_intensity, use_pixelate)

                st.success(f"Done. {num_faces} face(s) detected and blurred.")
                col1, col2 = st.columns(2)
                with col1:
                    st.image(cv2.cvtColor(cv2.imread(input_path), cv2.COLOR_BGR2RGB), caption="Original")
                with col2:
                    st.image(cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB), caption="Blurred")

                out_path = os.path.join(TEMP_DIR, "blurred_output.png")
                cv2.imwrite(out_path, result_frame)
                with open(out_path, "rb") as f:
                    st.download_button("Download Blurred Photo", f, file_name="blurred_photo.png", mime="image/png")
            except Exception as e:
                st.error(f"Something went wrong while processing the photo: {e}")
            finally:
                cleanup_temp_files(input_path)

    else:
        uploaded = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])
        if uploaded and st.button("Blur Faces in Video"):
            input_path = save_uploaded_file(uploaded)
            silent_output_path = os.path.join(TEMP_DIR, "auto_blur_silent.mp4")
            final_output_path = os.path.join(TEMP_DIR, "auto_blur_final.mp4")

            progress_bar = st.progress(0.0, text="Processing video...")
            try:
                process_video_auto_blur(
                    input_path,
                    silent_output_path,
                    blur_intensity=blur_intensity,
                    use_pixelate=use_pixelate,
                    progress_callback=lambda p: progress_bar.progress(p, text=f"Processing video... {int(p * 100)}%"),
                )

                with st.spinner("Merging original audio track..."):
                    has_audio = merge_audio_with_video(input_path, silent_output_path, final_output_path)

                progress_bar.progress(1.0, text="Done!")
                st.success(
                    "Video processed successfully"
                    + (" with original audio retained." if has_audio else " (no audio track found in source).")
                )

                st.video(final_output_path)
                with open(final_output_path, "rb") as f:
                    st.download_button("Download Blurred Video", f, file_name="blurred_video.mp4", mime="video/mp4")
            except MemoryError:
                st.error("Ran out of memory processing this video. Try a shorter clip or lower resolution.")
            except Exception as e:
                st.error(f"Something went wrong while processing the video: {e}")
            finally:
                cleanup_temp_files(input_path, silent_output_path)


def run_mode_selective(blur_intensity, use_pixelate):
    st.header("Mode 2 — Selective Blur")
    st.write("Scan a video for unique faces, choose who stays visible, and only the rest get blurred.")

    uploaded = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"], key="selective_uploader")

    if uploaded is None:
        return

    # Persist the uploaded video path across Streamlit reruns via session_state.
    if "selective_video_path" not in st.session_state or st.session_state.get("selective_video_name") != uploaded.name:
        st.session_state.selective_video_path = save_uploaded_file(uploaded)
        st.session_state.selective_video_name = uploaded.name
        st.session_state.pop("face_db", None)  # reset scan results for a newly uploaded video

    input_path = st.session_state.selective_video_path

    col_a, col_b = st.columns(2)
    with col_a:
        sample_rate = st.slider("Scan every Nth frame (lower = more thorough, slower)", 5, 60, 15)
    with col_b:
        tolerance = st.slider("Face match sensitivity (lower = stricter matching)", 0.3, 0.7, 0.5, 0.01)

    if st.button("Step 1: Scan Video for Unique Faces"):
        progress_bar = st.progress(0.0, text="Scanning video for faces...")
        try:
            db = extract_unique_faces(
                input_path,
                sample_every_n_frames=sample_rate,
                tolerance=tolerance,
                progress_callback=lambda p: progress_bar.progress(p, text=f"Scanning... {int(p * 100)}%"),
            )
            st.session_state.face_db = db
            progress_bar.progress(1.0, text="Scan complete!")

            if not db.faces:
                st.warning("No faces were detected in this video. Try lowering 'Scan every Nth frame'.")
            else:
                st.success(f"Found {len(db.faces)} unique face(s). Select who to blur below.")
        except Exception as e:
            st.error(f"Something went wrong while scanning the video: {e}")

    if "face_db" in st.session_state and st.session_state.face_db.faces:
        db = st.session_state.face_db
        st.subheader("Step 2: Choose Faces to Blur")

        selected_ids = []
        cols = st.columns(4)
        for i, face in enumerate(db.faces):
            with cols[i % 4]:
                st.image(cv2.cvtColor(face.thumbnail, cv2.COLOR_BGR2RGB), caption=f"Person {face.face_id}")
                blur_this = st.checkbox(f"Blur Person {face.face_id}", key=f"blur_{face.face_id}")
                if blur_this:
                    selected_ids.append(face.face_id)

        st.markdown("---")
        st.subheader("Step 3: Generate Final Video")

        if st.button("Apply Selective Blur & Render Video"):
            if not selected_ids:
                st.warning("No faces selected — nothing will be blurred. Select at least one face above.")
            else:
                silent_output_path = os.path.join(TEMP_DIR, "selective_blur_silent.mp4")
                final_output_path = os.path.join(TEMP_DIR, "selective_blur_final.mp4")
                progress_bar = st.progress(0.0, text="Rendering video...")
                try:
                    process_video_selective(
                        input_path,
                        silent_output_path,
                        db,
                        selected_ids,
                        tolerance=tolerance,
                        blur_intensity=blur_intensity,
                        progress_callback=lambda p: progress_bar.progress(p, text=f"Rendering... {int(p * 100)}%"),
                    )

                    with st.spinner("Merging original audio track..."):
                        has_audio = merge_audio_with_video(input_path, silent_output_path, final_output_path)

                    progress_bar.progress(1.0, text="Done!")
                    st.success("Selective blur applied successfully.")

                    st.video(final_output_path)
                    with open(final_output_path, "rb") as f:
                        st.download_button("Download Final Video", f, file_name="selective_blur_video.mp4", mime="video/mp4")
                except MemoryError:
                    st.error("Ran out of memory rendering this video. Try a shorter clip or lower resolution.")
                except Exception as e:
                    st.error(f"Something went wrong while rendering the video: {e}")
                finally:
                    cleanup_temp_files(silent_output_path)


if __name__ == "__main__":
    main()
