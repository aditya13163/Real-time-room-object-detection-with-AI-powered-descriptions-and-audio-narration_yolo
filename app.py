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

# --- Configuration ---
MODEL_PATH_STR = 'yolo_room_detection_result/cpu_run12/weights/best.pt' # Make sure this path is correct

# --- TTS Engine Setup ---
tts_lock = threading.Lock()
tts_engine = None # Initialize as None
try:
    tts_engine = pyttsx3.init()
    if tts_engine: # Check if init() returned a valid engine object
        tts_engine.setProperty('rate', 150)  # Speed of speech
        tts_engine.setProperty('volume', 0.9) # Volume (0.0 to 1.0)
    else:
        # This case handles if pyttsx3.init() returns None without raising an exception
        st.warning("TTS engine initialization returned None. Voice output will be disabled.")
except Exception as e:
    st.error(f"Error initializing TTS engine: {e}. Voice output will be disabled.")
    tts_engine = None # Ensure tts_engine is None if initialization fails

last_spoken_text_success = "" # Store only successfully spoken text

def speak_text(text_to_speak, force_speak=False):
    """
    Speaks the given text using pyttsx3 in a separate thread.
    Uses a lock to ensure thread-safe TTS operations.
    The problematic internal 'isBusy' check has been removed.
    """
    global last_spoken_text_success

    if not tts_engine:
        print(f"TTS Engine not available. Cannot speak: '{text_to_speak}'")
        return

    def speaking_thread_function():
        global last_spoken_text_success
        try:
            # tts_lock ensures that only one thread executes this block at a time for the TTS engine.
            with tts_lock:
                # Avoid rapidly repeating the exact same successfully spoken phrase unless forced
                if not force_speak and text_to_speak == last_spoken_text_success:
                    # print(f"Skipping repeated successfully spoken text: {text_to_speak}") # Optional debug
                    return

                current_text_to_speak = text_to_speak # Use a local copy for this thread's execution

                tts_engine.say(current_text_to_speak)
                tts_engine.runAndWait() # Blocks this thread until speech is done

                # If speech was successful, update the last successfully spoken text
                last_spoken_text_success = current_text_to_speak

        except RuntimeError as e:
            # This can happen if runAndWait is called while engine is in a bad state or already processing
            # in a way that runAndWait cannot handle (e.g., driver issues).
            print(f"TTS RuntimeError (possibly engine busy or bad state) for text '{text_to_speak}': {e}")
        except Exception as e:
            print(f"Error in TTS speaking thread for text '{text_to_speak}': {e}")

    # Start the speaking action in a new daemon thread
    thread = threading.Thread(target=speaking_thread_function)
    thread.daemon = True # Allows main program to exit even if this thread is still running
    thread.start()


# --- Object Descriptions (from your provided script) ---
OBJECT_DESCRIPTIONS = {
    'backpack': "A backpack is a bag carried on one's back, typically made of cloth or leather with straps over the shoulders.",
    'bottle-a': "A bottle is a container with a neck that is narrower than the body, used for storing liquids.",
    'bottle-b': "A second type of bottle, possibly different in shape or size from bottle-a.",
    'bowl': "A bowl is a round, deep dish used for preparing or serving food.",
    'casserole': "A casserole is a large, deep dish used both in the oven and as a serving vessel.",
    'chair': "A chair is a piece of furniture designed for sitting, typically with four legs and a back.",
    'cup': "A cup is a small open container used for drinking, usually with a handle.",
    'fork': "A fork is a utensil with prongs used for eating or serving food.",
    'frigo': "A refrigerator is a cooling appliance used to preserve food at low temperatures.",
    'glass': "A glass is a container, often made of glass, typically used for drinking.",
    'handbag': "A handbag is a small bag used to carry personal items, typically carried by women.",
    'iphone': "An iPhone is a line of smartphones designed and marketed by Apple Inc.",
    'knife': "A knife is a tool with a cutting edge or blade, used for cutting.",
    'lamp': "A lamp is a device that produces light.",
    'laptop': "A laptop is a portable computer suitable for use while traveling.",
    'macbook': "A MacBook is a brand of Macintosh laptop computers by Apple Inc.",
    'micro-ondes': "A microwave oven is an electric oven that heats and cooks food by exposing it to electromagnetic radiation.",
    'oldphone': "An old phone refers to earlier models of telephones, possibly rotary or early mobile phones.",
    'paperbag': "A paper bag is a bag made of paper, usually used for carrying goods.",
    'plate': "A plate is a flat dish for holding food.",
    'smartphone': "A smartphone is a mobile phone with advanced features and internet access.",
    'sofa': "A sofa is a long upholstered seat with a back and arms, for two or more people.",
    'spoon': "A spoon is a utensil with a shallow bowl on a handle, used for eating or stirring.",
    'table': "A table is a piece of furniture with a flat top and one or more legs.",
    'washmachine': "A washing machine is an appliance used to wash laundry."
}

# Mapping model class names to more "speakable" names if needed
CLASS_NAME_MAPPINGS = {
    "bottle-a": "bottle",
    "bottle-b": "bottle",
    "frigo": "refrigerator",
    "micro-ondes": "microwave oven",
    "washmachine": "washing machine",
}

def get_object_info(class_name_from_model):
    """Gets a speakable name and description for a given model class name."""
    speakable_name = CLASS_NAME_MAPPINGS.get(class_name_from_model, class_name_from_model).replace('-', ' ')
    description = OBJECT_DESCRIPTIONS.get(class_name_from_model, f"A {speakable_name} has been detected.")
    return speakable_name, description

# --- YOLO Model Loading ---
@st.cache_resource # Cache the model to load only once per session
def load_yolo_model(model_path_str):
    model_path = Path(model_path_str)
    if not model_path.exists():
        error_message = f"CRITICAL: Model weights file not found at {model_path.resolve()}"
        st.error(error_message)
        speak_text("Error: Model file not found.", force_speak=True)
        return None
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = YOLO(model_path) # Initialize model
        # model.to(device) # This might be implicitly handled by YOLO or can be done before inference
        st.success(f"YOLOv8 model loaded successfully from '{model_path.name}'. It will run on {device.upper()}.")
        speak_text(f"YOLO model loaded. It will use the {device}.", force_speak=True)
        return model, device # Return model and device
    except Exception as e:
        error_message = f"Error loading YOLO model: {e}"
        st.error(error_message)
        speak_text("An error occurred while loading the YOLO model.", force_speak=True)
        return None, None # Return None for both if loading fails

# --- Process Image/Frame and Detect Objects ---
def process_image_for_detection(image_np, model, device, confidence_threshold=0.35):
    """
    Processes an image, performs object detection, annotates the image,
    and prepares information for display and TTS.
    Returns: annotated_image_np, list of detected_object_infos
    """
    if model is None:
        st.error("Model is not loaded. Cannot perform detection.")
        return image_np, []

    annotated_image = image_np.copy()
    detected_objects_info_list = []

    try:
        # Perform inference on the specified device
        results = model(annotated_image, conf=confidence_threshold, device=device, verbose=False)

        if results and results[0] and results[0].boxes:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])

                class_name_from_model = model.names[cls_id] if model.names and cls_id < len(model.names) else f"ClassID:{cls_id}"
                speakable_name, description = get_object_info(class_name_from_model)

                speak_text(speakable_name) # Announce detected object's name

                detected_objects_info_list.append({
                    "id": f"{speakable_name}_{x1}_{y1}",
                    "name": speakable_name,
                    "confidence": conf,
                    "description": description,
                    "box_coords": (x1, y1, x2, y2)
                })

                label = f"{speakable_name} ({conf:.2f})"
                color = (0, 255, 0) # Green
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated_image, (x1, y1 - h - 10), (x1 + w, y1 - 5), color, -1)
                cv2.putText(annotated_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)

    except Exception as e:
        st.error(f"Error during object detection: {e}")
        speak_text("An error occurred during detection.")

    return annotated_image, detected_objects_info_list

# --- Streamlit App UI ---
st.set_page_config(layout="wide", page_title="YOLOv8 Object Detection & Voice Assistant")

st.title("👁️ YOLOv8 Room Object Detection with Voice Assistant 🔊")
st.markdown("""
Welcome! This app uses a custom-trained YOLOv8 model to detect objects in images or from your webcam.
It will announce the names of detected objects and you can click to hear their descriptions.
""")
st.markdown(f"**Model in use:** `{Path(MODEL_PATH_STR).name}`")
st.markdown("---")

# Load the model (cached)
model, inference_device = load_yolo_model(MODEL_PATH_STR)

if model and inference_device:
    col1, col2 = st.columns([2, 1])

    with col2:
        st.subheader("⚙️ Detection Controls")
        confidence_slider = st.slider("Detection Confidence Threshold", 0.10, 0.95, 0.35, 0.05)

        st.subheader("🎙️ Voice Output")
        if tts_engine:
            st.info("Text-to-speech is enabled. Ensure your speakers are on.")
        else:
            st.warning("Text-to-speech is disabled due to an initialization error or was not available.")

        st.subheader("📖 Object Descriptions")
        descriptions_placeholder = st.container()

    with col1:
        st.subheader("🖼️ Input Source")
        input_source = st.radio("Select Input:", ["Upload Image", "Webcam"], horizontal=True, key="input_source_radio")
        image_display_placeholder = st.empty()

        if input_source == "Upload Image":
            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp", "webp"])

            if uploaded_file is not None:
                try:
                    pil_image = Image.open(uploaded_file).convert("RGB")
                    image_np = np.array(pil_image)
                    speak_text("Processing uploaded image.", force_speak=True)
                    annotated_image, detected_objects = process_image_for_detection(image_np, model, inference_device, confidence_slider)
                    image_display_placeholder.image(annotated_image, caption="Processed Image with Detections", use_column_width=True)

                    with descriptions_placeholder:
                        descriptions_placeholder.empty() # Clear previous
                        if detected_objects:
                            st.markdown("---")
                            st.markdown(f"**Found {len(detected_objects)} object(s):**")
                            for obj_info in detected_objects:
                                st.markdown(f"**{obj_info['name']}** (Confidence: {obj_info['confidence']:.2f})")
                                if st.button(f"🔊 Hear description for {obj_info['name']}", key=f"speak_desc_{obj_info['id']}"):
                                    speak_text(f"This is a {obj_info['name']}. {obj_info['description']}", force_speak=True)
                                st.info(obj_info['description'])
                        else:
                            st.info("No objects detected above the set confidence threshold.")
                            speak_text("No objects detected in the image.")
                except Exception as e:
                    st.error(f"Error processing uploaded image: {e}")
                    speak_text("Error processing the uploaded image.")

        elif input_source == "Webcam":
            # Session state to manage webcam run status
            if 'run_webcam' not in st.session_state:
                st.session_state.run_webcam = False

            if st.checkbox("Start Webcam Detection", key="webcam_checkbox"):
                if not st.session_state.run_webcam:
                    st.session_state.run_webcam = True
                    speak_text("Starting webcam detection.", force_speak=True)
            else: # If checkbox is unchecked
                if st.session_state.run_webcam:
                    st.session_state.run_webcam = False # Stop webcam
                    speak_text("Webcam detection stopped by user.", force_speak=True)


            if st.session_state.run_webcam:
                st.info("Webcam detection is active. Uncheck the box to stop.")
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    st.error("Could not open webcam. Please check permissions and connections.")
                    speak_text("Could not open webcam.")
                    st.session_state.run_webcam = False # Reset flag
                else:
                    while st.session_state.run_webcam: # Check flag each iteration
                        ret, frame = cap.read()
                        if not ret:
                            st.warning("Failed to grab frame from webcam. Stream might have ended.")
                            speak_text("Webcam stream ended or failed.")
                            st.session_state.run_webcam = False # Stop on failure
                            break

                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        annotated_frame, detected_objects_live = process_image_for_detection(frame_rgb, model, inference_device, confidence_slider)
                        image_display_placeholder.image(annotated_frame, caption="Live Webcam Feed with Detections", use_column_width=True)

                        with descriptions_placeholder:
                            descriptions_placeholder.empty() # Clear previous frame's live detections
                            if detected_objects_live:
                                st.markdown("---")
                                st.markdown(f"**Live Detections ({len(detected_objects_live)}):**")
                                for obj_info in detected_objects_live:
                                    st.write(f"- {obj_info['name']} ({obj_info['confidence']:.2f})")
                            else:
                                st.markdown("*(No objects detected in current frame)*")
                        time.sleep(0.05) # Small delay
                    cap.release()
                    if not st.session_state.run_webcam: # If loop exited because flag became false
                         image_display_placeholder.empty() # Clear webcam image
                         descriptions_placeholder.empty()
                         st.info("Webcam detection stopped.")
            else:
                image_display_placeholder.info("Webcam is off. Check the 'Start Webcam Detection' box to begin.")
else:
    st.error("YOLOv8 model could not be loaded. Application cannot proceed with detection.")
    st.markdown(f"Please ensure the model file is correctly placed at: `{Path(MODEL_PATH_STR).resolve()}`")

st.markdown("---")
st.markdown("Object Detection App | Powered by Ultralytics YOLOv8 & Streamlit")