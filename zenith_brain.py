import threading
import queue
import time
import cv2
import numpy as np
import concurrent.futures as cf
import joblib
import os
import tensorflow as tf
from config import setup_logging, MODEL_DIR

logger = setup_logging("zenith.brain")
from tensorflow import keras
from tensorflow.keras import layers
from collections import deque, Counter
import mediapipe as mp
from biomechanical_features import (
    extract_biomechanical_features, StabilityTracker,
    compute_pose_quality_score, get_deviations, LABEL_TO_PROFILE
)

# TF Config
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
try:
    tf.config.set_visible_devices([], 'GPU')
except Exception:
    pass
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

class ZenithBrain:
    """
    The Thinking Engine (Async Worker).
    Handles heavy ML inference (MediaPipe, Classifier, VAE) and Flow Calculation.
    """
    def __init__(self):
        self.input_queue = queue.Queue(maxsize=1) 
        self.output_queue = queue.Queue(maxsize=1)
        self.running = True
        
        # --- MODELS ---
        self.load_classifier()
        self.load_vae()
        self.init_mediapipe()
        
        # --- FLOW STATE ---
        self.prev_landmarks_array = None
        self.prev_velocity = 0.0
        self.current_flow_score = 100.0
        self.prev_time = 0.0
        # Smoothing Buffer
        self.flow_buffer = deque(maxlen=7)

        # --- BIOMECHANICAL FEATURES ---
        self.stability_tracker = StabilityTracker(buffer_size=15)
        
        # --- WORKER ---
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()

    def load_classifier(self):
        self.clf_ok = False
        self.pose_classifier = None
        CLF_PATH = str(MODEL_DIR / "zenith_pose_classifier.pkl")
        if os.path.exists(CLF_PATH):
            try:
                self.pose_classifier = joblib.load(CLF_PATH)
                self.clf_ok = True
                logger.info("Classifier loaded")
            except Exception as e:
                logger.error(f"Classifier load error: {e}")

    def load_vae(self):
        self.vae_ok = False
        self.encoder = None
        self.decoder = None
        ENC_W_PATH = str(MODEL_DIR / "zenith_encoder_weights.weights.h5")
        DEC_W_PATH = str(MODEL_DIR / "zenith_decoder_weights.weights.h5")
        if os.path.exists(ENC_W_PATH) and os.path.exists(DEC_W_PATH):
            try:
                self.encoder, self.decoder = self.build_vae_model()
                self.encoder.load_weights(ENC_W_PATH)
                self.decoder.load_weights(DEC_W_PATH)
                self.vae_ok = True
                logger.info("VAE loaded")
            except Exception as e:
                logger.error(f"VAE load error: {e}")

    def build_vae_model(self, input_dim=132, latent_dim=16):
        class Sampling(layers.Layer):
            def call(self, inputs):
                z_mean, z_log_var = inputs
                eps = tf.keras.backend.random_normal(shape=(tf.shape(z_mean)[0], tf.shape(z_mean)[1]))
                return z_mean + tf.exp(0.5 * z_log_var) * eps
        enc_in = keras.Input(shape=(input_dim,))
        x = layers.Dense(64, activation="relu")(enc_in)
        z_mean = layers.Dense(latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        enc = keras.Model(enc_in, [z_mean, z_log_var, z], name="encoder")
        z_in = keras.Input(shape=(latent_dim,))
        x = layers.Dense(64, activation="relu")(z_in)
        dec_out = layers.Dense(input_dim, activation="sigmoid")(x)
        dec = keras.Model(z_in, dec_out, name="decoder")
        return enc, dec

    def init_mediapipe(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def landmarks_to_flat(self, lms):
        arr = np.array([[lm.x,lm.y,lm.z,lm.visibility] for lm in lms.landmark], dtype=np.float32)
        return arr.flatten()[None,:]

    def landmarks_to_biomechanical(self, lms):
        """Extract 30 biomechanical features from MediaPipe landmarks.
        Returns (1, 30) float32 array suitable for classification."""
        bio_feats = extract_biomechanical_features(lms, self.stability_tracker)
        return bio_feats[None, :].astype(np.float32)
    
    def calculate_flow(self, curr_flat, prev_flat, dt, prev_v):
        if prev_flat is None or dt <= 0: return 100.0, 0.0
        diff = curr_flat - prev_flat
        dist = np.linalg.norm(diff)
        velocity = dist / dt
        acceleration = abs(velocity - prev_v) / dt
        # Punishment factor
        punishment = acceleration * 15.0
        score = 100.0 - punishment
        return float(np.clip(score, 0, 100)), velocity

    def calculate_bio_flow(self, curr_bio, prev_bio, dt):
        """Calculate flow from biomechanical feature angular velocities.
        More meaningful than raw landmark displacement — measures joint
        angular change rate rather than pixel movement."""
        if prev_bio is None or dt <= 0:
            return 100.0
        # Angular velocity: rate of change of joint angles (first 16 features)
        angle_diff = np.abs(curr_bio[0, :16] - prev_bio[0, :16])
        # Denormalize to degrees/sec
        angular_velocity = (angle_diff * 180.0) / dt
        # Jerk proxy: high angular velocity = jerky
        mean_angular_vel = float(np.mean(angular_velocity))
        # Score: smooth motion has low angular velocity
        # Normalize: 0°/s = perfect stillness (100), 50°/s = moderate (50), 100°/s = jerky (0)
        score = 100.0 - np.clip(mean_angular_vel, 0, 100)
        return float(score)

    def _worker_loop(self):
        POSE_NAMES = ["Chair","Downward Dog","Extended Side Angle","High Lunge","Mountain Pose","Plank","Tree","Triangle","Warrior II"]
        INT_TO_POSE = {i:n for i,n in enumerate(POSE_NAMES)}
        
        while self.running:
            try:
                frame, timestamp = self.input_queue.get(timeout=0.1)
            except queue.Empty:
                continue
                
            result = {
                'pose_landmarks': None,
                'label': None,
                'vae_q': None,
                'vae_recon': None,
                'flow_score': self.current_flow_score, # Return last known if fail
                'velocity': self.prev_velocity,
                'bio_features': None,
                'bio_quality': None,
                'bio_deviations': None,
            }
            
            try:
                # 1. MediaPipe
                res = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                result['pose_landmarks'] = res.pose_landmarks
                
                dt = timestamp - self.prev_time if self.prev_time > 0 else 0
                self.prev_time = timestamp

                if res.pose_landmarks:
                    feats = self.landmarks_to_flat(res.pose_landmarks)

                    # 2. Classifier (with confidence threshold)
                    if self.clf_ok:
                        proba = self.pose_classifier.predict_proba(feats)
                        max_conf = float(np.max(proba))
                        pred_idx = int(np.argmax(proba))
                        if max_conf >= 0.6:  # Only label if >= 60% confident
                            result['label'] = INT_TO_POSE.get(pred_idx, "Unknown")
                            result['confidence'] = max_conf
                        else:
                            result['label'] = None  # Not a recognized pose

                    # 2b. Biomechanical features (parallel path)
                    bio_feats = self.landmarks_to_biomechanical(res.pose_landmarks)
                    result['bio_features'] = bio_feats
                    if result['label'] and result['label'] in LABEL_TO_PROFILE:
                        profile_key = LABEL_TO_PROFILE[result['label']]
                        result['bio_quality'] = compute_pose_quality_score(
                            bio_feats[0], profile_key
                        )
                        result['bio_deviations'] = get_deviations(
                            bio_feats[0], profile_key
                        )

                    # 3. VAE Quality
                    if self.vae_ok:
                         z_mean, _, z = self.encoder.predict(feats, verbose=0)
                         recon = self.decoder.predict(z, verbose=0)
                         mse = float(np.mean((feats - recon) ** 2))
                         q = 100.0 * (1.0 - (mse - 0.0005) / (0.006 - 0.0005))
                         result['vae_q'] = float(np.clip(q, 0, 100))
                         result['vae_recon'] = recon
                    
                    # 4. Flow Calculation
                    if self.prev_landmarks_array is not None and dt > 0:
                        raw_score, velocity = self.calculate_flow(feats, self.prev_landmarks_array, dt, self.prev_velocity)
                        
                        # Add to buffer
                        self.flow_buffer.append(raw_score)
                        
                        # Smooth (Average)
                        smoothed_score = sum(self.flow_buffer) / len(self.flow_buffer)
                        
                        # Exponential Decay on top for continuity
                        alpha_flow = 0.2
                        self.current_flow_score = (alpha_flow * smoothed_score) + ((1-alpha_flow) * self.current_flow_score)
                        self.prev_velocity = velocity
                        
                        result['flow_score'] = self.current_flow_score
                        result['velocity'] = velocity

                    self.prev_landmarks_array = feats

            except Exception as e:
                logger.error(f"Worker error: {e}")
                
            if not self.output_queue.empty():
                try:
                    self.output_queue.get_nowait()
                except queue.Empty:
                    pass
            self.output_queue.put(result)
            self.input_queue.task_done()

    def process_async(self, frame, timestamp):
        """Non-blocking process request."""
        if not self.input_queue.full():
            self.input_queue.put((frame, timestamp))
            
    def get_latest_result(self):
        """Returns None if no new result, else result dict."""
        if not self.output_queue.empty():
            try:
                return self.output_queue.get_nowait()
            except queue.Empty:
                return None
        return None
