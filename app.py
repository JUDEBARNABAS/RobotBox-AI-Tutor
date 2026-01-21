import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from streamlit_mic_recorder import mic_recorder
import google.generativeai as genai
from google import genai as live_genai # New SDK for Live API features
import cv2
import os
import threading
import io
import base64
from PIL import Image
from dotenv import load_dotenv

# 1. Setup & API Configuration
load_dotenv()
api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")

# Initialize both SDKs (One for standard config, one for Live features)
genai.configure(api_key=api_key)
client = live_genai.Client(api_key=api_key, http_options={'api_version': 'v1beta'})

st.set_page_config(page_title="RobotBox AI Lab", layout="wide", page_icon="🤖")

# 2. AI Tutor Socratic Instruction
socratic_instruction = """
# ROLE
    You are the RobotBox AI Tutor, an expert engineering mentor inspired by Sal Khan's Socratic teaching style. 
    Your goal is to guide students (ages 8-16) through building robotics projects delivered in their monthly RobotBox.

    # SOCRATIC PRINCIPLES
    1. NEVER GIVE ANSWERS: If a student says "Where does this wire go?", do not say "Pin 5." Instead, ask "Looking at the motor driver, which pins are labeled for power?"
    2. VALIDATE VISION: You can see their workspace via the camera. If you see a mistake (e.g., a loose battery), say "I'm looking at your battery pack... does that red wire look like it's tucked in all the way?"
    3. THINK ALOUD: Explain the 'why.' Instead of "Check the code," say "I'm thinking about how the robot knows when to stop. What sensor would help it 'see' a wall?"
    4. ENCOURAGE MISTAKES: If they fail, treat it as data. "That's a great 'oops'! What did the robot do differently than we expected?"

    # TONE
    Encouraging, curious, and professional but fun. Use analogies to explain complex electronics (e.g., "Electricity is like water flowing through pipes").
    """

# 3. Camera Buffer (Thread-safe)
lock = threading.Lock()
img_container = {"img": None}

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    with lock:
        img_container["img"] = img
    return frame

# 4. UI Layout
st.title("🤖 RobotBox AI Lab")
st.caption("Now with Native Voice & Vision")
st.markdown("---")

col_left, col_right = st.columns([1.5, 1])

with col_left:
    st.subheader("📷 Engineering Feed")
    webrtc_streamer(
        key="robotbox-live",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        ),
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col_right:
    st.subheader("💬 AI Interaction")
    
    # Audio Recorder Component
    st.write("Hold to talk to your Tutor:")
    audio_data = mic_recorder(
        start_prompt="🎤 Start Speaking",
        stop_prompt="🛑 Stop & Send",
        key='recorder'
    )

    if audio_data:
        # 5. Process Multi-Modal Input
        with st.spinner("Tutor is thinking..."):
            try:
                # Capture the current camera frame
                with lock:
                    current_frame = img_container["img"]
                
                # Prepare parts for Gemini
                parts = []
                
                # Add Audio
                parts.append({
                    "mime_type": "audio/wav",
                    "data": base64.b64encode(audio_data['bytes']).decode()
                })
                
                # Add Image if available
                if current_frame is not None:
                    img_rgb = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(img_rgb)
                    img_byte_arr = io.BytesIO()
                    pil_img.save(img_byte_arr, format='JPEG')
                    parts.append({
                        "mime_type": "image/jpeg",
                        "data": base64.b64encode(img_byte_arr.getvalue()).decode()
                    })

                # 6. Call the Native Audio Model
                # This model returns BOTH text and audio bytes
                response = client.models.generate_content(
                    model="gemini-2.5-flash-native-audio-preview-12-2025",
                    contents=parts,
                    config={
                        "system_instruction": socratic_instruction,
                        "response_modalities": ["AUDIO", "TEXT"]
                    }
                )

                # 7. Output: Text and Audio
                # Display text
                if response.text:
                    st.chat_message("assistant").write(response.text)
                
                # Play Audio Response
                # Note: The model returns raw PCM or WAV data depending on config
                audio_part = next((p for p in response.candidates[0].content.parts if p.inline_data), None)
                if audio_part:
                    st.audio(audio_part.inline_data.data, format="audio/wav")
                    st.success("🔈 Audio response generated!")

            except Exception as e:
                st.error(f"Brain Error: {e}")