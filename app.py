import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from google import genai
from google.genai import types
import cv2
import threading
import PIL.Image
import io
import os
import asyncio
import base64
from dotenv import load_dotenv

# 1. Setup
load_dotenv()
api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=api_key, http_options={'api_version': 'v1beta'})

# Model from your script
MODEL_ID = "gemini-2.5-flash-native-audio-preview-12-2025"

st.set_page_config(page_title="RobotBox Live Lab", layout="wide")

# 2. Socratic System Instruction
SYSTEM_INSTRUCTION = """
# ROLE
You are the RobotBox AI Tutor, an expert engineering mentor inspired by Sal Khan's Socratic teaching style. 

# SOCRATIC PRINCIPLES
1. NEVER GIVE ANSWERS: If a student says "Where does this wire go?", ask them to identify labels.
2. VALIDATE VISION: Use the camera feed to spot errors. Mention colors or shapes.
3. THINK ALOUD: Explain the 'why' using analogies.
"""

# 3. Vision Buffer (Thread-safe)
lock = threading.Lock()
media_state = {"img": None}

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    # Fix the "blue tint" by converting BGR to RGB (from your script)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    with lock:
        media_state["img"] = img_rgb
    return frame

# 4. UI Layout
st.title("🎙️ RobotBox Live AI Lab")

col_video, col_chat = st.columns([1.5, 1])

with col_video:
    st.subheader("📷 Robot Workspace")
    webrtc_ctx = webrtc_streamer(
        key="robotbox-live",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        ),
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": True},
        async_processing=True,
    )

with col_chat:
    st.subheader("💬 Tutor Session")
    
    if webrtc_ctx.state.playing:
        if st.button("Connect with Tutor"):
            with st.status("Initializing Live WebSocket...") as status:
                # This mimics your async run() loop but inside the Streamlit context
                async def run_live_session():
                    async with client.aio.live.connect(model=MODEL_ID, config={
                        "system_instruction": SYSTEM_INSTRUCTION,
                        "response_modalities": ["AUDIO"]
                    }) as session:
                        status.update(label="Tutor is listening!", state="running")
                        
                        while webrtc_ctx.state.playing:
                            with lock:
                                frame = media_state["img"]
                            
                            if frame is not None:
                                # Process frame for Gemini
                                pil_img = PIL.Image.fromarray(frame)
                                buffer = io.BytesIO()
                                pil_img.save(buffer, format="JPEG")
                                
                                # Send Frame
                                await session.send(input={
                                    "mime_type": "image/jpeg",
                                    "data": base64.b64encode(buffer.getvalue()).decode()
                                })
                            
                            # Listen for Audio Responses
                            async for response in session.receive():
                                if response.data:
                                    st.audio(response.data, format="audio/wav")
                            
                            await asyncio.sleep(1.0) # Prevent rate limiting

                asyncio.run(run_live_session())
    else:
        st.info("Start the camera feed to begin your session.")