import cv2
import mediapipe as mp
import numpy as np
import time
import streamlit as st
import tempfile
import shutil
from gtts import gTTS
import io
import pygame

# Initialize MediaPipe Pose module
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialize Pygame for audio playback
pygame.init()

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

# Function to generate and play TTS audio
def speak(text):
    tts = gTTS(text, lang='en')
    audio_file = io.BytesIO()
    tts.write_to_fp(audio_file)
    audio_file.seek(0)
    
    pygame.mixer.music.load(audio_file)
    pygame.mixer.music.play()
    
    # Wait for the audio to finish playing
    while pygame.mixer.music.get_busy():
        time.sleep(0.1)

def process_video(input_file, output_file):
    cap = cv2.VideoCapture(input_file)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        # No drawing or text added to the frame
        out.write(frame)

    cap.release()
    out.release()

# Streamlit UI
st.title('Biceps Curl Posture Correction')
option = st.selectbox("Choose input method", ["Use Camera", "Upload Video"])

if option == "Upload Video":
    uploaded_file = st.file_uploader("Upload a video", type=['mp4', 'mov'])
    
    if uploaded_file:
        # Create temporary files for input and output
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_input_file:
            temp_input_file.write(uploaded_file.read())
            temp_input_file_path = temp_input_file.name

        temp_output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_output_file_path = temp_output_file.name

        # Process the uploaded video
        process_video(temp_input_file_path, temp_output_file_path)

        # Provide a download button for the processed video
        with open(temp_output_file_path, 'rb') as f:
            st.download_button(label="Download Processed Video", data=f, file_name='processed_biceps_curl.mp4', mime='video/mp4')

        # Ensure files are properly closed before deletion
        temp_input_file.close()
        temp_output_file.close()

        # Clean up temporary files
        try:
            shutil.os.remove(temp_input_file_path)
            shutil.os.remove(temp_output_file_path)
        except OSError as e:
            st.error(f"Error removing temporary files: {e}")

elif option == "Use Camera":
    stframe = st.empty()
    cap = cv2.VideoCapture(0)
    t = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        feedback_text = ""

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Define keypoints for the biceps curl exercise for both arms
            arms = {
                'left': {
                    'shoulder': (int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * frame.shape[1]),
                                 int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * frame.shape[0])),
                    'elbow': (int(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x * frame.shape[1]),
                             int(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y * frame.shape[0])),
                    'wrist': (int(landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x * frame.shape[1]),
                             int(landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y * frame.shape[0]))
                },
                'right': {
                    'shoulder': (int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * frame.shape[1]),
                                 int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * frame.shape[0])),
                    'elbow': (int(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x * frame.shape[1]),
                             int(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y * frame.shape[0])),
                    'wrist': (int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x * frame.shape[1]),
                             int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y * frame.shape[0]))
                }
            }

            # Calculate angle for the biceps curl for both arms
            angles = {arm: calculate_angle(data['shoulder'], data['elbow'], data['wrist']) for arm, data in arms.items()}

            # Set feedback conditions and colors for both arms
            for arm, angle in angles.items():
                if angle < 100:
                    feedback_text += f"{arm.capitalize()} arm: Fully relax your arm\n"
                    speak("Relax")
                elif angle > 160:
                    feedback_text += f"{arm.capitalize()} arm: Fully flex your arm\n"
                    speak("Flex")

        # Display the frame without lines or text
        stframe.image(frame, channels="BGR", use_column_width=True)

        # Display feedback text in a separate placeholder
        feedback_placeholder = st.empty()
        feedback_placeholder.markdown(f'<p style="font-size:24px;">{feedback_text}</p>', unsafe_allow_html=True)



    cap.release()
