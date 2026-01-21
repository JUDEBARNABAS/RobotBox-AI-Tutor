import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import google.generativeai as genai
import cv2
import av
import os
import threading
from dotenv import load_dotenv
from PIL import Image

# 1. Setup & API Configuration
load_dotenv()
api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

st.set_page_config(page_title="RobotBox AI Tutor", layout="wide")

# 2. Socratic Configuration
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

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash", # Using the faster 2.0 model
    system_instruction=socratic_instruction
)

# 3. Thread-safe Frame Buffer
# This captures the most recent image from the webcam stream
lock = threading.Lock()
img_container = {"img": None}

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    with lock:
        img_container["img"] = img
    return frame

# 4. Interface Layout
st.title("🤖 RobotBox Lab: Socratic Tutor")
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Live Engineering Feed")
    # WebRTC allows the camera to work on the deployed URL
    webrtc_ctx = webrtc_streamer(
        key="robotbox-feed",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        ),
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    st.info("Click 'Start' above to open your webcam.")

with col2:
    st.subheader("Socratic Chat")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about your build..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        # Get the latest frame for the AI to see
        with lock:
            latest_img = img_container["img"]

        with st.chat_message("assistant"):
            with st.spinner("Analyzing your workspace..."):
                try:
                    contents = [prompt]
                    if latest_img is not None:
                        # Convert BGR to RGB for Gemini
                        img_rgb = cv2.cvtColor(latest_img, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(img_rgb)
                        contents.append(pil_img)
                    
                    response = model.generate_content(contents)
                    st.markdown(response.text)
                    st.session_state.messages.append({"role": "assistant", "content": response.text})
                except Exception as e:
                    st.error(f"Brain Error: {e}")