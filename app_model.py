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
from PIL import Image

# --- Configuration ---
MODEL_PATH_STR = 'yolo_room_detection_result/cpu_run12/weights/best.pt'
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", "")
TEMP_AUDIO_DIR = "temp_audio"
os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)

# --- Initialize Gemini ---
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest')
else:
    gemini_model = None
    st.warning("Gemini API key not found. Using basic descriptions.")


# --- Audio System ---
class AudioPlayer:
    def __init__(self):
        self.queue = queue.Queue()
        self.lock = threading.Lock()
        self.active = True
        self.speed_factor = 1.5
        self.worker = threading.Thread(target=self._process_queue, daemon=True)
        self.worker.start()
        self.last_play_time = 0
        self.cooldown = 1.0

    def _text_to_speech(self, text):
        try:
            tts = gTTS(text=text, lang='en', slow=False)
            temp_file = os.path.join(TEMP_AUDIO_DIR, f"temp_{time.time()}.mp3")
            tts.save(temp_file)
            data, fs = sf.read(temp_file)
            os.remove(temp_file)
            return (data, fs)
        except Exception as e:
            st.error(f"TTS Error: {str(e)}")
            return None

    def add_alert(self, text):
        current_time = time.time()
        if current_time - self.last_play_time > self.cooldown:
            audio_data = self._text_to_speech(text)
            if audio_data:
                self.queue.put(audio_data)
                self.last_play_time = current_time

    def _process_queue(self):
        while self.active:
            try:
                if not self.queue.empty():
                    data, fs = self.queue.get()
                    with self.lock:
                        sd.play(data, int(fs * self.speed_factor))
                        sd.wait()
            except Exception as e:
                st.error(f"Audio Playback Error: {str(e)}")
            time.sleep(0.05)

    def stop(self):
        self.active = False
        self.worker.join()
        sd.stop()


# --- AI Description Generator ---
class AIDescriber:
    def __init__(self):
        self.cache = {}
        self.last_request_time = 0
        self.request_cooldown = 0.5

    def describe(self, class_name):
        if class_name in self.cache:
            return self.cache[class_name]

        if not GOOGLE_API_KEY:
            return f"{class_name} detected"

        current_time = time.time()
        if current_time - self.last_request_time < self.request_cooldown:
            return f"{class_name} detected"

        try:
            self.last_request_time = current_time
            response = gemini_model.generate_content(
                f"Create a 3-5 word audible alert for a {class_name} for vision assistance. "
                "Include location if possible. Example: 'Person ahead', 'Dog on left'",
                generation_config=genai.types.GenerationConfig(
                    temperature=0.4,
                    max_output_tokens=20
                )
            )
            description = response.text.strip()
            self.cache[class_name] = description
            return description
        except Exception as e:
            st.error(f"Gemini API Error: {str(e)}")
            return f"{class_name} detected"


# --- Object Detector ---
class ObjectDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.describer = AIDescriber()
        self.last_detection_time = {}

    def process_frame(self, frame, conf_thresh):
        detections = []
        annotated = frame.copy()

        try:
            results = self.model(frame, conf=conf_thresh, device=self.device)

            if results and results[0].boxes:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = box.conf[0].item()
                    cls_id = box.cls[0].item()
                    class_name = self.model.names.get(cls_id, f"object-{cls_id}")

                    position = self._get_position_description(x1, y1, x2, y2, frame.shape)
                    desc = self.describer.describe(class_name)

                    detections.append({
                        "class": class_name,
                        "conf": conf,
                        "desc": f"{desc} {position}",
                        "box": [x1, y1, x2, y2],
                        "timestamp": time.time()
                    })

                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated, f"{class_name} {conf:.2f}",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        except Exception as e:
            st.error(f"Detection Error: {str(e)}")

        return annotated, detections

    def _get_position_description(self, x1, y1, x2, y2, frame_shape):
        height, width = frame_shape[:2]
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        vertical_pos = "top" if center_y < height / 3 else "bottom" if center_y > 2 * height / 3 else ""
        horizontal_pos = "left" if center_x < width / 3 else "right" if center_x > 2 * width / 3 else "center"

        return f"at {horizontal_pos} {vertical_pos}".strip()


# --- Streamlit App ---
def main():
    st.set_page_config(
        page_title="AI Vision Assistant",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("👁️ AI Vision Assistant with Voice Alerts")

    # Initialize systems
    if 'detector' not in st.session_state:
        st.session_state.detector = ObjectDetector(Path(MODEL_PATH_STR))
    if 'audio' not in st.session_state:
        st.session_state.audio = AudioPlayer()

    # Initialize session state variables
    if 'webcam_active' not in st.session_state:
        st.session_state.webcam_active = False
    if 'last_detections' not in st.session_state:
        st.session_state.last_detections = []
    if 'detection_history' not in st.session_state:
        st.session_state.detection_history = []

    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Settings")
        conf_thresh = st.slider("Confidence Threshold", 0.1, 0.9, 0.35, 0.05)
        st.session_state.audio.speed_factor = st.slider("Voice Speed", 1.0, 3.0, 1.5, 0.1)
        st.session_state.audio.cooldown = st.slider("Alert Cooldown (sec)", 0.5, 5.0, 1.5, 0.5)

        st.markdown("---")
        st.write(f"**Hardware**: {st.session_state.detector.device.upper()}")
        st.write(f"**AI Status**: {'✅ Active' if GOOGLE_API_KEY else '❌ Disabled'}")

        if st.button("▶️ Test Voice System"):
            st.session_state.audio.add_alert("Voice system is working properly")

    # Main interface
    col1, col2 = st.columns([3, 1])

    with col1:
        source = st.radio("Input Source", ["📷 Image", "🎥 Webcam"], horizontal=True)

        if "Image" in source:
            uploaded = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
            if uploaded:
                image = Image.open(uploaded).convert("RGB")
                frame = np.array(image)

                with st.spinner("Processing image..."):
                    processed, detections = st.session_state.detector.process_frame(frame, conf_thresh)
                    st.image(processed, channels="RGB", use_column_width=True)

                # Update detections
                st.session_state.last_detections = detections
                if detections:
                    st.session_state.detection_history.extend(detections)
                    for obj in detections:
                        st.session_state.audio.add_alert(obj['desc'])

        elif "Webcam" in source:
            if st.button("⏯️ Start Webcam" if not st.session_state.webcam_active else "⏹️ Stop Webcam"):
                st.session_state.webcam_active = not st.session_state.webcam_active

            if st.session_state.webcam_active:
                cap = cv2.VideoCapture(0)
                frame_placeholder = st.empty()

                try:
                    while st.session_state.webcam_active:
                        ret, frame = cap.read()
                        if not ret:
                            st.warning("Failed to capture frame")
                            break

                        processed, detections = st.session_state.detector.process_frame(
                            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                            conf_thresh
                        )

                        frame_placeholder.image(processed, channels="RGB")

                        # Update detections
                        if detections:
                            st.session_state.last_detections = detections
                            st.session_state.detection_history.extend(detections)
                            for obj in detections:
                                if obj['class'] not in st.session_state.detector.last_detection_time or \
                                        time.time() - st.session_state.detector.last_detection_time[obj['class']] > 2.0:
                                    st.session_state.audio.add_alert(obj['desc'])
                                    st.session_state.detector.last_detection_time[obj['class']] = time.time()

                        time.sleep(0.05)

                finally:
                    cap.release()
                    st.session_state.webcam_active = False
                    frame_placeholder.empty()

    # Detection panel - ALWAYS VISIBLE
    with col2:
        st.subheader("🔍 Detected Objects")

        if st.session_state.last_detections:
            for obj in st.session_state.last_detections:
                with st.expander(f"{obj['class']} ({obj['conf']:.2f})"):
                    st.write(f"**Description**: {obj['desc']}")
                    st.write(f"**Position**: {obj['box']}")
                    st.write(f"**Time**: {time.strftime('%H:%M:%S', time.localtime(obj['timestamp']))}")
        else:
            st.info("No objects detected currently")

        st.markdown("---")
        st.subheader("📜 Detection History")

        if st.session_state.detection_history:
            # Show last 10 detections (most recent first)
            history_to_show = st.session_state.detection_history[-10:][::-1]
            for obj in history_to_show:
                st.write(f"**{obj['class']}** ({(obj['conf']):.2f}): {obj['desc']}")
        else:
            st.info("No detection history yet")

    # Cleanup
    st.session_state.audio.stop()


if __name__ == "__main__":
    main()