import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from streamlit_mic_recorder import mic_recorder
from google import genai
from google.genai import types
import cv2
import os
import threading
import io
import PIL.Image
from dotenv import load_dotenv

# 1. Setup & API Configuration
load_dotenv()
# Streamlit Cloud will use st.secrets; local will use .env
api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")

# Use the latest GenAI Client for Live features
client = genai.Client(api_key=api_key, http_options={'api_version': 'v1beta'})
MODEL_ID = "gemini-2.0-flash" # Stable version for 2026

st.set_page_config(page_title="RobotBox AI Lab", layout="wide", page_icon="🤖")

# 2. Socratic System Instructions
socratic_instruction = """
# ROLE
You are the RobotBox AI Tutor, an expert engineering mentor. 
# SOCRATIC PRINCIPLES
1. NEVER GIVE ANSWERS. If a student asks where a wire goes, ask them to identify the labels on the board.
2. VALIDATE VISION: Use the camera feed to spot errors. Reference specific colors or shapes.
3. BE CONCISE: Since you are responding with audio, keep your explanations brief and engaging.
"""

# 3. Vision Buffer (Thread-safe)
lock = threading.Lock()
shared_frames = {"img": None}

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    with lock:
        shared_frames["img"] = img
    return frame

# 4. UI Layout
st.title("🤖 RobotBox Lab: AI Tutor")
st.caption("Native Voice & Vision Enabled")
st.write("---")

col_video, col_chat = st.columns([1.5, 1])

with col_video:
    st.subheader("📷 Engineering Feed")
    webrtc_streamer(
        key="robotbox-vision",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        ),
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    st.info("💡 Tip: Use headphones to prevent the AI from hearing itself!")

with col_chat:
    st.subheader("💬 AI Interaction")
    
    # Audio Recorder Component
    st.write("Hold to talk to your Tutor:")
    audio_data = mic_recorder(
        start_prompt="🎤 Start Speaking",
        stop_prompt="🛑 Stop & Send",
        key='robotbox_recorder'
    )

    # Initialize chat history for text display
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Process Input
    if audio_data:
        with st.spinner("Tutor is analyzing..."):
            try:
                # Capture the current frame
                with lock:
                    current_frame = shared_frames["img"]
                
                # Build Multimodal Parts using strict google.genai.types
                parts = []
                
                # Add Audio Part
                parts.append(
                    types.Part.from_bytes(
                        data=audio_data['bytes'], 
                        mime_type="audio/wav"
                    )
                )
                
                # Add Vision Part
                if current_frame is not None:
                    img_rgb = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
                    pil_img = PIL.Image.fromarray(img_rgb)
                    img_byte_arr = io.BytesIO()
                    pil_img.save(img_byte_arr, format='JPEG')
                    
                    parts.append(
                        types.Part.from_bytes(
                            data=img_byte_arr.getvalue(), 
                            mime_type="image/jpeg"
                        )
                    )

                # Generate Content
                response = client.models.generate_content(
                    model=MODEL_ID,
                    contents=parts,
                    config=types.GenerateContentConfig(
                        system_instruction=socratic_instruction,
                        response_modalities=["AUDIO", "TEXT"]
                    )
                )

                # Output Text
                if response.text:
                    with st.chat_message("assistant"):
                        st.markdown(response.text)
                
                # Output Audio
                # Finding the audio part in the response
                for part in response.candidates[0].content.parts:
                    if part.inline_data:
                        st.audio(part.inline_data.data, format="audio/wav")

            except Exception as e:
                st.error(f"Brain Error: {e}")

    # Display message history (Optional)
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])