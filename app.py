# === ZENith MVP - Final Production Version ===
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# --- Page Configuration (Must be the first Streamlit command) ---
st.set_page_config(layout="wide")

# --- CORE PARAMETERS & MODEL DEFINITIONS ---

# This single value controls the sensitivity of the quality score.
# It represents the maximum reconstruction error for a "perfect" pose.
# A smaller number (e.g., 0.005) makes the scoring stricter.
MAX_ACCEPTABLE_MSE = 0.01

# Custom Keras classes are required for loading the saved VAE model.
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction = self.decoder(z)
        return reconstruction

# --- MODEL ARCHITECTURE & LOADING ---
# The model architecture must be defined to load the weights correctly.

@st.cache_resource
def load_all_models():
    # Model Dimensions (must match training)
    latent_dim = 16
    input_dim = 132

    # Build Encoder Architecture
    encoder_inputs = keras.Input(shape=(input_dim,))
    x_enc = layers.Dense(64, activation="relu")(encoder_inputs)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x_enc)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x_enc)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    # Build Decoder Architecture
    latent_inputs = keras.Input(shape=(latent_dim,))
    x_dec = layers.Dense(64, activation="relu")(latent_inputs)
    decoder_outputs = layers.Dense(input_dim, activation="sigmoid")(x_dec)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

    # Load the trained weights
    classifier = joblib.load('zenith_pose_classifier.pkl')
    encoder.load_weights('zenith_encoder_weights.weights.h5')
    decoder.load_weights('zenith_decoder_weights.weights.h5')
    
    return classifier, encoder, decoder

pose_classifier, quality_encoder, quality_decoder = load_all_models()

# --- MEDIAPIPE SETUP & LABELS ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# This list MUST match the alphabetical order used during training.
pose_names = ['Chair', 'Crescent', 'Downward Dog', 'Extended Side Angle', 'High Lunge', 
              'Mountain Pose', 'Plank', 'Tree', 'Triangle', 'Warrior II']
int_to_pose_labels = {i: name for i, name in enumerate(pose_names)}

# --- HELPER FUNCTION: ANGLE CALCULATION ---
def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Mid point
    c = np.array(c)  # End point
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

# --- STREAMLIT UI ---
st.title('ZENith $ZEN^{ith}$ - Live AI Coaching Demo')
st.write("Perform a yoga flow. The AI will identify poses and analyze your form in real-time.")

# --- VIDEO PROCESSING CLASS ---
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def transform(self, frame):
        image = frame.to_ndarray(format="bgr24")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        current_pose = "..."
        quality_score = 0

        if results.pose_landmarks:
            try:
                landmarks_list = results.pose_landmarks.landmark
                landmarks = [[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks_list]
                flat_keypoints = np.array(landmarks).flatten().reshape(1, -1)
                
                prediction = pose_classifier.predict(flat_keypoints)
                current_pose = int_to_pose_labels.get(prediction[0], "Unrecognized")
                
                # Quality Score Calculation
                z_mean, _, z = quality_encoder.predict(flat_keypoints, verbose=0)
                reconstructed_keypoints = quality_decoder.predict(z, verbose=0)
                mse = np.mean((flat_keypoints - reconstructed_keypoints)**2)
                if not np.isnan(mse):
                    error_ratio = np.clip(mse / MAX_ACCEPTABLE_MSE, 0, 1)
                    quality_score = 100 * (1 - error_ratio)
                else:
                    quality_score = 0
                
                # Get coordinates for angle calculation
                coords = {lm: [landmarks[lm.value][0], landmarks[lm.value][1]] for lm in mp_pose.PoseLandmark}

                # Calculate all angles
                angles = {
                    "l_knee": calculate_angle(coords[mp_pose.PoseLandmark.LEFT_HIP], coords[mp_pose.PoseLandmark.LEFT_KNEE], coords[mp_pose.PoseLandmark.LEFT_ANKLE]),
                    "r_knee": calculate_angle(coords[mp_pose.PoseLandmark.RIGHT_HIP], coords[mp_pose.PoseLandmark.RIGHT_KNEE], coords[mp_pose.PoseLandmark.RIGHT_ANKLE]),
                    "l_elbow": calculate_angle(coords[mp_pose.PoseLandmark.LEFT_SHOULDER], coords[mp_pose.PoseLandmark.LEFT_ELBOW], coords[mp_pose.PoseLandmark.LEFT_WRIST]),
                    "r_elbow": calculate_angle(coords[mp_pose.PoseLandmark.RIGHT_SHOULDER], coords[mp_pose.PoseLandmark.RIGHT_ELBOW], coords[mp_pose.PoseLandmark.RIGHT_WRIST]),
                    "l_shoulder": calculate_angle(coords[mp_pose.PoseLandmark.LEFT_HIP], coords[mp_pose.PoseLandmark.LEFT_SHOULDER], coords[mp_pose.PoseLandmark.LEFT_ELBOW]),
                    "r_shoulder": calculate_angle(coords[mp_pose.PoseLandmark.RIGHT_HIP], coords[mp_pose.PoseLandmark.RIGHT_SHOULDER], coords[mp_pose.PoseLandmark.RIGHT_ELBOW])
                }
                
                # Display angles on the video feed
                h, w, _ = image_bgr.shape
                for joint, angle in angles.items():
                    joint_name = joint.split('_')[1].upper() # e.g., KNEE
                    landmark_name = f"LEFT_{joint_name}" if "l_" in joint else f"RIGHT_{joint_name}"
                    coord = tuple(np.multiply(coords[mp_pose.PoseLandmark[landmark_name]], [w,h]).astype(int))
                    cv2.putText(image_bgr, str(int(angle)), coord,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2, cv2.LINE_AA)
                
            except Exception as e:
                current_pose = "Processing..."

        # Drawing and Display logic
        score_color = (0, 0, 255); # Red
        if quality_score > 85: score_color = (0, 255, 0) # Green
        elif quality_score > 60: score_color = (0, 255, 255) # Yellow

        mp_drawing.draw_landmarks(image_bgr, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                  mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
        
        cv2.rectangle(image_bgr, (0,0), (320, 80), (245, 117, 66), -1)
        cv2.putText(image_bgr, f'POSE: {current_pose}', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image_bgr, 'QUALITY:', (10,65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        bar_x, bar_y, bar_w, bar_h = 140, 48, 150, 20
        cv2.rectangle(image_bgr, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (0,0,0), -1)
        cv2.rectangle(image_bgr, (bar_x, bar_y), (bar_x + int(bar_w * (quality_score / 100)), bar_y + bar_h), score_color, -1)

        return image_bgr

webrtc_streamer(
    key="zenith-mvp-final",
    video_processor_factory=VideoTransformer,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
)
