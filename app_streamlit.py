# app.py
import streamlit as st
from ultralytics import YOLO
import cv2
import torch
from pathlib import Path
import numpy as np
import time
import google.generativeai as genai
from gtts import gTTS
import sounddevice as sd
import soundfile as sf
import queue
import threading
import os
import platform
from PIL import Image

# --- Configuration ---
MODEL_PATH_STR = 'yolo_room_detection_result/cpu_run12/weights/best.pt'  # Ensure this path is correct for your environment
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", "")
TEMP_AUDIO_DIR = "temp_audio"
os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)

# --- Initialize Gemini ---
if GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest')
    except Exception as e:
        st.error(f"Failed to initialize Gemini: {e}")
        gemini_model = None
else:
    gemini_model = None
    st.warning("Google API Key not found. AI descriptions will be basic class names.")


# --- Enhanced Audio System (TTS in Worker Thread) ---
class AudioPlayer:
    def __init__(self):
        self.text_queue = queue.Queue()
        self.active = True
        self.worker_thread = threading.Thread(target=self._process_audio_queue, daemon=True)
        self.speed_factor = 1.5  # Default playback speed
        self.worker_thread.start()

    def _generate_speech_data(self, text):
        """Generates audio data from text."""
        try:
            tts = gTTS(text=text, lang='en', slow=False)
            temp_file_path = os.path.join(TEMP_AUDIO_DIR, f"audio_{time.time_ns()}.mp3")
            tts.save(temp_file_path)
            data, samplerate = sf.read(temp_file_path)
            os.remove(temp_file_path)
            return data, samplerate
        except Exception as e:
            print(f"Error in TTS generation for '{text}': {e}")
            st.toast(f"TTS Error for: {text[:30]}...", icon="🔊")
            return None, None

    def _play_audio_data(self, data, samplerate):
        """Plays audio data."""
        try:
            sd.play(data, int(samplerate * self.speed_factor))
            sd.wait()  # Waits for the current sound to finish before playing the next
        except Exception as e:
            print(f"Error playing audio: {e}")
            st.toast("Audio playback error.", icon="🔊")

    def _process_audio_queue(self):
        """Worker thread: gets text, generates speech, plays it."""
        while self.active:
            try:
                text_to_speak = self.text_queue.get(timeout=0.1)
                if text_to_speak is None:  # Sentinel for stopping
                    self.text_queue.task_done()
                    break

                audio_data, fs = self._generate_speech_data(text_to_speak)
                if audio_data is not None and fs is not None:
                    self._play_audio_data(audio_data, fs)
                self.text_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in audio processing worker: {e}")
                if 'text_to_speak' in locals() and text_to_speak is not None:  # type: ignore
                    self.text_queue.task_done()  # Ensure task is marked done

    def announce(self, text):
        """Adds text to the queue to be announced."""
        if text:  # Add text to queue if it's not empty
            self.text_queue.put(text)

    def set_speed(self, factor):
        self.speed_factor = factor

    def stop(self):
        print("Attempting to stop AudioPlayer...")
        self.active = False
        self.text_queue.put(None)  # Signal worker to exit
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=3)  # Wait for worker
        sd.stop()  # Stop any ongoing playback
        # Clean up remaining temp files if any (optional, as they are usually deleted)
        # for f in Path(TEMP_AUDIO_DIR).glob("*.mp3"):
        #     try:
        #         os.remove(f)
        #     except OSError:
        #         pass # ignore if deletion fails
        print("AudioPlayer stopped.")

    def test_audio_system(self):
        self.announce("Audio system is functioning correctly.")


# --- AI Description System with Caching ---
class AIDescriber:
    def __init__(self):
        self.cache = {}

    def describe(self, class_name):
        if not gemini_model:  # If Gemini model failed to initialize or no API key
            return f"{class_name} detected."

        if class_name in self.cache:
            return self.cache[class_name]

        try:
            prompt = f"Describe a '{class_name}' in  22 words for an audio alert (e.g., 'A comfortable chair for sitting', or 'A modern laptop computer'). Keep it concise, around 3-7 words."
            response = gemini_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.4,
                    max_output_tokens=30
                )
            )
            description = response.text.strip().replace("*", "").replace("\n", " ")
            if not description or len(description) < 3:  # Basic validation
                description = f"A {class_name}."
            self.cache[class_name] = description
            return description
        except Exception as e:
            print(f"Gemini API Error for '{class_name}': {str(e)}")
            st.toast(f"Gemini error for {class_name}", icon="🧠")
            fallback_description = f"{class_name} detected, AI description unavailable."
            self.cache[class_name] = fallback_description  # Cache fallback to avoid retries
            return fallback_description


# --- Object Detection System ---
class ObjectDetector:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.describer = AIDescriber()
        self._load_model()

    def _load_model(self):
        try:
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            print(f"YOLO model loaded successfully on {self.device} from {self.model_path}")
        except Exception as e:
            st.error(f"Error loading YOLO model from {self.model_path}: {e}")
            print(f"Error loading YOLO model: {e}")
            self.model = None

    def process_frame(self, frame, conf_thresh):
        detections_list = []
        annotated_frame = frame.copy()

        if not self.model:
            st.error("Object detection model not loaded. Cannot process frame.")
            return annotated_frame, detections_list

        try:
            results = self.model(frame, conf=conf_thresh, device=self.device, verbose=False)

            if results and results[0].boxes:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = box.conf[0].item()
                    cls_id = int(box.cls[0].item())  # Ensure cls_id is an int for dictionary keys
                    class_name = self.model.names.get(cls_id, f"Object-{cls_id}")

                    description = self.describer.describe(class_name)
                    detections_list.append({
                        "class": class_name,
                        "conf": conf,
                        "desc": description,
                        "bbox": (x1, y1, x2, y2)
                    })

                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label_text = f"{class_name} {conf:.2f}"
                    cv2.putText(annotated_frame, label_text,
                                (x1, y1 - 10 if y1 > 20 else y1 + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        except Exception as e:
            st.error(f"Detection Error: {str(e)}")
            print(f"Detection runtime error: {e}")

        return annotated_frame, detections_list


# --- Streamlit UI ---
def main():
    st.set_page_config(
        page_title="AI Object Analyst",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("👁️ Real-Time Object Detection with AI Narration 🔊")

    # Initialize systems
    # Check if model path exists
    model_file = Path(MODEL_PATH_STR)
    if not model_file.exists():
        st.error(f"YOLO Model file not found at {MODEL_PATH_STR}. Please check the path.")
        st.stop()

    detector = ObjectDetector(model_file)
    audio_player = AudioPlayer()  # Renamed instance for clarity

    # Session state
    if 'webcam_active' not in st.session_state:
        st.session_state.webcam_active = False
    if 'last_announced_objects' not in st.session_state:
        st.session_state.last_announced_objects = set()
    if 'last_announcement_time' not in st.session_state:
        st.session_state.last_announcement_time = time.time()

    # Control Panel
    with st.sidebar:
        st.header("⚙️ Controls")
        conf_thresh = st.slider("Confidence Threshold", 0.10, 0.90, 0.35, 0.05)

        playback_speed = st.slider("Playback Speed", 1.0, 3.0, 1.5, 0.1,
                                   help="Adjust the speed of audio announcements.")
        audio_player.set_speed(playback_speed)

        if st.button("Test Audio System 🎤"):
            audio_player.test_audio_system()

        st.markdown("---")
        st.info("Ensure your speakers are on!")
        if not GOOGLE_API_KEY:
            st.warning("No Google API Key. Descriptions will be basic.")
        if detector.device == 'cpu':
            st.info("Running on CPU. Detection may be slower.")
        else:
            st.success("Running on GPU (CUDA).")

    # Main interface
    col1, col2 = st.columns([2, 1])  # Adjusted column ratio

    with col1:
        st.subheader("Input Source")
        source_option = st.radio("Choose input:", ["🖼️ Image Upload", "📹 Live Webcam"], horizontal=True,
                                 label_visibility="collapsed")

        if source_option == "🖼️ Image Upload":
            st.session_state.webcam_active = False  # Ensure webcam is off
            uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

            if uploaded_file:
                image = Image.open(uploaded_file).convert("RGB")
                frame_np = np.array(image)

                with st.spinner("Processing image..."):
                    processed_frame, detections = detector.process_frame(frame_np, conf_thresh)

                st.image(processed_frame, channels="RGB", caption="Processed Image")

                # Announce and display detections for image
                if detections:
                    st.session_state.last_announced_objects.clear()  # Clear previous for new image
                    unique_descriptions = set()
                    for obj in detections:
                        unique_descriptions.add(obj['desc'])

                    for desc_to_announce in sorted(list(unique_descriptions)):  # Sort for consistent order
                        audio_player.announce(desc_to_announce)

                    with col2:
                        st.subheader("🔍 Detections")
                        if not detections:
                            st.info("No objects detected above the threshold.")
                        for i, obj in enumerate(detections):
                            with st.expander(f"{i + 1}. {obj['class']} ({obj['conf']:.2f})", expanded=True):
                                st.write(f"Description: {obj['desc']}")
                else:
                    with col2:
                        st.subheader("🔍 Detections")
                        st.info("No objects detected in the uploaded image.")


        elif source_option == "📹 Live Webcam":
            webcam_button_text = "⏹️ Stop Webcam" if st.session_state.webcam_active else "▶️ Start Webcam"
            if st.button(webcam_button_text):
                st.session_state.webcam_active = not st.session_state.webcam_active
                if not st.session_state.webcam_active:  # If stopping
                    st.session_state.last_announced_objects.clear()
                    st.info("Webcam stopped.")
                else:
                    st.info("Webcam starting...")

            frame_placeholder = st.empty()
            detections_placeholder = col2.empty()  # Placeholder for live detections in col2

            if st.session_state.webcam_active:
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    st.error("Could not open webcam.")
                    st.session_state.webcam_active = False
                else:
                    st.toast("Webcam activated!", icon="📹")

                try:
                    while st.session_state.webcam_active and cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            st.warning("Failed to grab frame from webcam. Stopping.")
                            st.session_state.webcam_active = False
                            break

                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        processed_frame, detections = detector.process_frame(frame_rgb, conf_thresh)
                        frame_placeholder.image(processed_frame, channels="RGB", caption="Live Webcam Feed")

                        current_detected_descs = {obj['desc'] for obj in detections}

                        # Announce new objects not recently announced (every ~3 seconds for a given object type)
                        # More sophisticated logic might be needed for continuous presence vs. new appearance
                        if detections and (
                                time.time() - st.session_state.last_announcement_time > 2.0):  # Announce new sets of objects every 2s
                            newly_seen_for_announcement = set()
                            for obj in detections:
                                # A simple way to avoid re-announcing the same thing too quickly
                                # This example announces all current unique descriptions if it's time to announce
                                newly_seen_for_announcement.add(obj['desc'])

                            if newly_seen_for_announcement:
                                for desc_to_announce in sorted(list(newly_seen_for_announcement)):
                                    audio_player.announce(desc_to_announce)
                                st.session_state.last_announcement_time = time.time()

                        # Update detection panel in column 2
                        with detections_placeholder.container():
                            st.subheader("🔴 Live Detections")
                            if detections:
                                for i, obj in enumerate(detections):
                                    st.markdown(f"**{i + 1}. {obj['class']}** (`{obj['conf']:.2f}`)")
                                    st.caption(f"🗣️: *{obj['desc']}*")
                                    st.divider()
                            else:
                                st.info("Scanning for objects...")

                        time.sleep(0.05)  # Small delay to allow other processes, adjust as needed

                finally:
                    if 'cap' in locals() and cap.isOpened():
                        cap.release()
                    frame_placeholder.empty()  # Clear image
                    with detections_placeholder.container():  # Clear detections display
                        st.info("Webcam feed stopped.")
                    st.session_state.webcam_active = False  # Ensure state is updated
                    print("Webcam released and placeholders cleared.")

    # This stop is called when the script finishes or Streamlit forces a rerun from the top.
    # For true "on browser close" cleanup, Streamlit doesn't offer robust server-side hooks easily.
    # The daemon thread for audio will stop when the main Streamlit process stops.
    # audio_player.stop() # This might be problematic with Streamlit's execution model.
    # Let daemon thread handle exit. Explicit stop can be for specific actions.


if __name__ == "__main__":
    main()
    # When the script ends (e.g. Ctrl+C in terminal), this part may or may not execute fully
    # depending on how Streamlit handles script termination.
    # The audio_player.stop() could be called here if running as a simple python script,
    # but in Streamlit, the app lifecycle is different.
    # The `atexit` module could be an option for cleanup if needed.
    # For now, relying on daemon thread for audio.