import threading
import queue
import time
import cv2
import numpy as np
import concurrent.futures as cf
import joblib
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from collections import deque, Counter
import mediapipe as mp

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
    Handles heavy ML inference (MediaPipe, Classifier, VAE) in a background thread.
    """
    def __init__(self):
        self.input_queue = queue.Queue(maxsize=1) # Only keep latest frame
        self.output_queue = queue.Queue(maxsize=1)
        self.running = True
        
        # --- MODELS ---
        self.load_classifier()
        self.load_vae()
        self.init_mediapipe()
        
        # --- WORKER ---
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()

    def load_classifier(self):
        self.clf_ok = False
        self.pose_classifier = None
        CLF_PATH = "zenith_pose_classifier.pkl"
        if os.path.exists(CLF_PATH):
            try:
                self.pose_classifier = joblib.load(CLF_PATH)
                self.clf_ok = True
            except Exception as e:
                print(f"Brain CLF Error: {e}")

    def load_vae(self):
        self.vae_ok = False
        self.encoder = None
        self.decoder = None
        ENC_W_PATH = "zenith_encoder_weights.weights.h5"
        DEC_W_PATH = "zenith_decoder_weights.weights.h5"
        if os.path.exists(ENC_W_PATH) and os.path.exists(DEC_W_PATH):
            try:
                self.encoder, self.decoder = self.build_vae_model()
                self.encoder.load_weights(ENC_W_PATH)
                self.decoder.load_weights(DEC_W_PATH)
                self.vae_ok = True
            except Exception as e:
                print(f"Brain VAE Error: {e}")

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

    def _worker_loop(self):
        POSE_NAMES = ["Chair","Downward Dog","Extended Side Angle","High Lunge","Mountain Pose","Plank","Tree","Triangle","Warrior II"]
        INT_TO_POSE = {i:n for i,n in enumerate(POSE_NAMES)}
        
        while self.running:
            try:
                frame = self.input_queue.get(timeout=0.1)
            except queue.Empty:
                continue
                
            result = {
                'pose_landmarks': None,
                'label': None,
                'vae_q': None,
                'vae_recon': None,
                'dream_recon': None
            }
            
            try:
                # 1. MediaPipe
                res = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                result['pose_landmarks'] = res.pose_landmarks
                
                if res.pose_landmarks:
                    feats = self.landmarks_to_flat(res.pose_landmarks)
                    
                    # 2. Classifier
                    if self.clf_ok:
                        pred = self.pose_classifier.predict(feats)
                        result['label'] = INT_TO_POSE.get(int(pred[0]), "Unknown")
                        
                    # 3. VAE Quality
                    if self.vae_ok:
                         z_mean, _, z = self.encoder.predict(feats, verbose=0)
                         recon = self.decoder.predict(z, verbose=0)
                         mse = float(np.mean((feats - recon) ** 2))
                         q = 100.0 * (1.0 - (mse - 0.0005) / (0.006 - 0.0005))
                         result['vae_q'] = float(np.clip(q, 0, 100))
                         result['vae_recon'] = recon

            except Exception as e:
                print(f"Brain Error: {e}")
                
            # Put result (clear old if exist for low latency)
            if not self.output_queue.empty():
                try:
                    self.output_queue.get_nowait()
                except queue.Empty:
                    pass
            self.output_queue.put(result)
            self.input_queue.task_done()

    def process_async(self, frame):
        """Non-blocking process request."""
        if not self.input_queue.full():
            self.input_queue.put(frame)
            
    def get_latest_result(self):
        """Returns None if no new result, else result dict."""
        if not self.output_queue.empty():
            try:
                return self.output_queue.get_nowait()
            except queue.Empty:
                return None
        return None
