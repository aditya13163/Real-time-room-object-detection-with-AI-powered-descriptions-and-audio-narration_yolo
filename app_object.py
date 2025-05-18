# app.py
import streamlit as st
from ultralytics import YOLO
import cv2
import torch
from pathlib import Path
import pyttsx3
import threading
from PIL import Image
import numpy as np
import time
import google.generativeai as genai
from collections import defaultdict

# --- Configuration ---
MODEL_PATH_STR = 'yolo_room_detection_result/cpu_run12/weights/best.pt'
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", "")  # Add to Streamlit secrets

# --- Initialize Gemini ---
genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel('models/gemini-1.5-flash-latest') if GOOGLE_API_KEY else None


# --- AI Context Generator ---
class AIContextGenerator:
    def __init__(self):
        self.description_cache = defaultdict(str)
        self.enabled = bool(GOOGLE_API_KEY)

    def get_position_context(self, bbox, image_size):
        """Determine object position relative to frame"""
        x_center = (bbox[0] + bbox[2]) / 2
        y_center = (bbox[1] + bbox[3]) / 2
        img_width, img_height = image_size

        vertical_pos = "top" if y_center < img_height / 3 else "bottom" if y_center > 2 * img_height / 3 else "middle"
        horizontal_pos = "left" if x_center < img_width / 3 else "right" if x_center > 2 * img_width / 3 else "center"
        return f"{vertical_pos}-{horizontal_pos}"

    def generate_description(self, class_name, bbox, image_size):
        """Generate AI-powered description using Gemini"""
        if not self.enabled:
            return ""

        cache_key = f"{class_name}_{self.get_position_context(bbox, image_size)}"

        if cache_key in self.description_cache:
            return self.description_cache[cache_key]

        try:
            position = self.get_position_context(bbox, image_size)
            prompt = f"""Generate a concise 2-sentence description of a {class_name} in a room environment.
            Include its potential use, location context ({position.replace('-', ' ')} area), 
            and one interesting fact. Be observational and factual."""

            response = gemini_model.generate_content(prompt)

            if response.prompt_feedback.block_reason:
                return f"A {class_name} is visible in the {position.replace('-', ' ')} area."

            description = response.text.strip()
            self.description_cache[cache_key] = description
            return description

        except Exception as e:
            print(f"Gemini Error: {str(e)}")
            return f"A {class_name} has been detected."


# Initialize AI Generator
ai_generator = AIContextGenerator()


# --- Voice Assistant ---
class VoiceAssistant:
    def __init__(self):
        self.engine = None
        self.lock = threading.Lock()
        self.last_message = ""
        self.init_engine()

    def init_engine(self):
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 150)
            self.engine.setProperty('volume', 0.9)
        except Exception as e:
            st.error(f"TTS Error: {str(e)}")

    def speak(self, text):
        if not self.engine or text == self.last_message:
            return

        def _speak():
            with self.lock:
                try:
                    self.engine.say(text)
                    self.engine.runAndWait()
                    self.last_message = text
                except RuntimeError:
                    self.init_engine()

        threading.Thread(target=_speak, daemon=True).start()


voice = VoiceAssistant()


# --- Model Loader ---
@st.cache_resource
def load_model(path):
    model_path = Path(path)
    if not model_path.exists():
        st.error(f"Model not found at {model_path.resolve()}")
        return None

    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = YOLO(model_path)
        model.to(device)
        return model, device
    except Exception as e:
        st.error(f"Model Error: {str(e)}")
        return None, None


# --- Detection Processor ---
def process_frame(frame, model, device, conf_thresh):
    detections = []
    if model is None:
        return frame, detections

    try:
        results = model(frame, conf=conf_thresh, device=device)
        h, w = frame.shape[:2]

        if results and results[0].boxes:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = box.conf[0].item()
                cls_id = box.cls[0].item()
                class_name = model.names.get(cls_id, f"object-{cls_id}")

                # Generate description
                description = ai_generator.generate_description(
                    class_name,
                    (x1, y1, x2, y2),
                    (w, h)
                )

                detections.append({
                    "class": class_name,
                    "conf": conf,
                    "desc": description,
                    "bbox": (x1, y1, x2, y2)
                })

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{class_name} {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Speak initial detection
                voice.speak(f"{class_name} detected")

    except Exception as e:
        st.error(f"Detection Error: {str(e)}")

    return frame, detections


# --- Streamlit UI ---
def main():
    st.set_page_config(
        page_title="AI Object Analyst",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("👁️ Smart Object Detection with Gemini 🔊")
    model, device = load_model(MODEL_PATH_STR)

    # Sidebar Controls
    with st.sidebar:
        st.header("Controls")
        conf_thresh = st.slider("Confidence Threshold", 0.1, 0.9, 0.35, 0.05)
        ai_enabled = st.checkbox("Enable AI Descriptions", value=ai_generator.enabled)
        tts_enabled = st.checkbox("Enable Voice Assistant", value=voice.engine is not None)

        if not GOOGLE_API_KEY:
            st.warning("Add GOOGLE_API_KEY to secrets.toml for AI descriptions")

    # Main Content
    col1, col2 = st.columns([3, 1])

    with col1:
        source = st.radio("Input Source", ["Image Upload", "Webcam"], horizontal=True)

        if source == "Image Upload":
            uploaded = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
            if uploaded:
                image = Image.open(uploaded).convert("RGB")
                frame = np.array(image)
                processed, detections = process_frame(frame, model, device, conf_thresh)
                st.image(processed, channels="RGB")

                with col2:
                    if detections:
                        st.subheader("AI Analysis")
                        for obj in detections:
                            with st.expander(f"{obj['class']} ({obj['conf']:.2f})"):
                                st.write(obj['desc'])
                                if st.button(f"🔊 Hear", key=f"btn_{obj['class']}"):
                                    voice.speak(obj['desc'])
                    else:
                        st.info("No objects detected")

        elif source == "Webcam":
            if st.button("Start Webcam"):
                cap = cv2.VideoCapture(0)
                placeholder = st.empty()
                stop_btn = st.button("Stop Webcam")

                while not stop_btn and cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    processed, detections = process_frame(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                        model,
                        device,
                        conf_thresh
                    )

                    placeholder.image(processed, channels="RGB")

                    # Update analysis panel
                    with col2:
                        st.subheader("Live Analysis")
                        if detections:
                            for obj in detections:
                                st.markdown(f"**{obj['class']}** ({obj['conf']:.2f})")
                                st.caption(obj['desc'])
                        else:
                            st.info("No objects in current frame")

                    time.sleep(0.1)

                cap.release()
                placeholder.empty()


if __name__ == "__main__":
    main()