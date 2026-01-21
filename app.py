import streamlit as st
import cv2
import google.generativeai as genai
import os
from dotenv import load_dotenv
from PIL import Image

# 1. Setup & API Configuration
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

st.set_page_config(page_title="RobotBox AI Tutor", layout="wide")

# 2. Define the Socratic Brain
# This instruction forces the AI to act as a mentor, not just an answer-bot.
socratic_instruction = """
You are the RobotBox AI Tutor, a friendly engineering mentor. 
Your goal is to guide students (ages 8-16) through building their robotics kits.
- NEVER give direct answers (e.g., don't say 'Connect the red wire to Pin 5').
- INSTEAD, ask guiding questions: 'What happens to the circuit if power doesn't have a path back to the battery?'
- If they are stuck, ask them to describe what they see or hold a part up to the camera.
- Use analogies like 'Electricity is like water flowing through pipes.'
"""

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    system_instruction=socratic_instruction
)

# 3. Sidebar for Settings
with st.sidebar:
    st.title("🤖 RobotBox Settings")
    kit_month = st.selectbox("Current Box", ["Month 1: The Voyager", "Month 2: Sentry Arm"])
    st.info(f"AI Tutor is now configured for {kit_month}")
    
    if st.button("Reset Session"):
        st.session_state.messages = []
        st.rerun()

# 4. Main Interface Layout
st.title("Welcome to your RobotBox Lab")
col1, col2 = st.columns([2, 1])

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Column 1: Live Video Feed
with col1:
    st.subheader("Live Engineering Feed")
    frame_placeholder = st.empty()
    stop_button = st.button("Stop Camera")
    
    cap = cv2.VideoCapture(0)
    while cap.isOpened() and not stop_button:
        ret, frame = cap.read()
        if not ret:
            st.error("Webcam not detected.")
            break
            
        # Convert BGR to RGB for Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB")
        
        # In a real "Live API" setup, you'd send frames to Gemini here. 
        # For now, we focus on the Socratic Chat interface.
        if stop_button:
            break
    cap.release()

# Column 2: Socratic Chat Interface
with col2:
    st.subheader("AI Tutor Chat")
    
    # Display Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat Input
    if user_input := st.chat_input("Ask a question about your build..."):
        # Add User Message to History
        st.chat_message("user").write(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Generate Socratic Response
        try:
            # We pass the full history for context
            chat_session = model.start_chat(history=[
                {"role": m["role"], "parts": [m["content"]]} for m in st.session_state.messages[:-1]
            ])
            response = chat_session.send_message(user_input)
            
            # Display Assistant Response
            with st.chat_message("assistant"):
                st.markdown(response.text)
            st.session_state.messages.append({"role": "assistant", "content": response.text})
            
        except Exception as e:
            st.error(f"Error connecting to Gemini: {e}")