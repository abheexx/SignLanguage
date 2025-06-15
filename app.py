import streamlit as st
import cv2
import numpy as np
from realtime_predict import SignLanguagePredictor
import torch
import os
from PIL import Image
import time

# Set page config
st.set_page_config(
    page_title="Sign Language Recognition",
    page_icon="🤟",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
    }
    .prediction-text {
        font-size: 24px;
        font-weight: bold;
        color: #2e7d32;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("🤟 Real-Time Sign Language Recognition")
st.markdown("""
    This application uses deep learning to recognize American Sign Language (ASL) gestures in real-time.
    The model combines CNN and LSTM architectures to capture both spatial and temporal features of sign language gestures.
""")

# Sidebar
st.sidebar.title("About")
st.sidebar.info("""
    This project was selected as the **Best Project among 50+ entries** in the AI/ML showcase.
    
    ### Features:
    - Real-time ASL gesture recognition
    - Deep learning architecture (CNN + LSTM)
    - Live prediction display
    - Support for multiple ASL signs
""")

# Initialize session state
if 'predictor' not in st.session_state:
    # Load model and class names
    model_path = "models/sign_language_model.pth"  # Update with your model path
    class_names = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
                  "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
    
    if os.path.exists(model_path):
        st.session_state.predictor = SignLanguagePredictor(model_path, class_names)
    else:
        st.error("Model file not found. Please ensure the model is in the correct location.")

# Main content
col1, col2 = st.columns(2)

with col1:
    st.header("Live Webcam Feed")
    # Create a placeholder for the webcam feed
    webcam_placeholder = st.empty()
    
    # Add a start/stop button
    if 'running' not in st.session_state:
        st.session_state.running = False
    
    if st.button("Start/Stop Webcam"):
        st.session_state.running = not st.session_state.running

with col2:
    st.header("Prediction")
    prediction_placeholder = st.empty()
    confidence_placeholder = st.empty()
    
    # Add a history of predictions
    st.subheader("Recent Predictions")
    history_placeholder = st.empty()

# Initialize webcam
if st.session_state.running:
    cap = cv2.VideoCapture(0)
    
    # Initialize prediction history
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []
    
    try:
        while st.session_state.running:
            # Process frame
            frame, prediction, confidence = st.session_state.predictor.process_video_stream(cap)
            
            if frame is not None:
                # Convert frame to RGB for display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Display webcam feed
                webcam_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
                
                # Display prediction
                if prediction:
                    prediction_placeholder.markdown(
                        f'<p class="prediction-text">Predicted Sign: {prediction}</p>',
                        unsafe_allow_html=True
                    )
                    confidence_placeholder.markdown(
                        f'<p class="prediction-text">Confidence: {confidence:.2%}</p>',
                        unsafe_allow_html=True
                    )
                    
                    # Update prediction history
                    st.session_state.prediction_history.append((prediction, confidence))
                    if len(st.session_state.prediction_history) > 5:
                        st.session_state.prediction_history.pop(0)
                    
                    # Display history
                    history_text = ""
                    for pred, conf in st.session_state.prediction_history:
                        history_text += f"- {pred}: {conf:.2%}\n"
                    history_placeholder.text(history_text)
            
            time.sleep(0.1)  # Add small delay to prevent high CPU usage
    
    finally:
        cap.release()
else:
    # Display placeholder when webcam is not running
    webcam_placeholder.info("Click 'Start/Stop Webcam' to begin recognition")
    prediction_placeholder.info("No prediction available")
    confidence_placeholder.info("No confidence score available")
    history_placeholder.info("No prediction history available")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>Built with ❤️ using Streamlit, PyTorch, and OpenCV</p>
    </div>
""", unsafe_allow_html=True) 