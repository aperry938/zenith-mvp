import os, time, sys, random
from collections import deque, Counter
import concurrent.futures as cf
import threading, queue

import cv2
import numpy as np
import av
import joblib
import mediapipe as mp

# --- TensorFlow ---
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# --- INTERNAL IMPORTS ---
try:
    from pose_foundations import PoseHeuristics
    from pose_sequencer import PoseSequencer
    from ui_overlay import ZenithUI
    from session_manager import SessionManager
    from data_harvester import DataHarvester
    from vision_client import VisionClient
except ImportError:
    print("Warning: Internal modules not found.")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
try:
    tf.config.set_visible_devices([], 'GPU')
except Exception:
    pass
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)


class ZenithCore:
    def __init__(self):
        self.stats_file = "zenith_stats.json"
        
        # --- STATE ---
        self.i = 0
        self.fps = 0.0
        self.t0 = time.time()
        self.last_label = None
        self.last_ok_ts = 0.0
        self.TTL = 1.5
        
        self.active_correction = None
        self.stability_start_time = None
        self.is_locked = False
        self.STABILITY_THRESHOLD = 3.0
        self.LOCKED_PHRASES = ["Locked.", "Solid.", "Perfect.", "Holding strong.", "That's it."]
        
        # flow
        self.prev_landmarks_array = None
        self.current_flow_score = 100.0
        self.prev_velocity = 0.0
        self.last_flow_msg_time = 0
        self.FLOW_MSG_DEBOUNCE = 10.0
        
        # tts
        self.tts_queue = queue.Queue()
        self.last_spoken_time = 0
        self.DEBOUNCE_SECONDS = 5.0
        self.start_tts_worker()
        
        # models
        self.load_classifier()
        self.load_vae()
        self.init_mediapipe()
        
        # dream state
        self.dream_z_current = None
        self.dream_z_target = None
        self.dream_t = 0.0
        self.DREAM_SPEED = 0.02
        
        # modules
        self.ui = ZenithUI()
        self.session = SessionManager()
        self.sequencer = PoseSequencer()
        self.harvester = DataHarvester()
        self.sage = VisionClient()
        
        self.latest_frame = None
        self.last_recon = None
        self.last_q = None

    def start_tts_worker(self):
        def tts_worker():
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.setProperty('rate', 160)
                while True:
                    text = self.tts_queue.get()
                    if text is None: break
                    try:
                        engine.say(text)
                        engine.runAndWait()
                    except Exception as e:
                        print(f"TTS Error: {e}")
                    self.tts_queue.task_done()
            except ImportError:
                 pass
        threading.Thread(target=tts_worker, daemon=True).start()

    def load_classifier(self):
        self.clf_ok = False
        self.pose_classifier = None
        self.POSE_NAMES = ["Chair","Downward Dog","Extended Side Angle","High Lunge","Mountain Pose","Plank","Tree","Triangle","Warrior II"]
        self.INT_TO_POSE = {i:n for i,n in enumerate(self.POSE_NAMES)}
        self.hist = deque(maxlen=15)
        
        CLF_PATH = "zenith_pose_classifier.pkl"
        if os.path.exists(CLF_PATH):
            try:
                self.pose_classifier = joblib.load(CLF_PATH)
                self.clf_ok = True
            except Exception as e:
                print(f"CLF Error: {e}")

    def load_vae(self):
        self.vae_ok = False
        self.encoder = None
        self.decoder = None
        self.vae_pool = cf.ThreadPoolExecutor(max_workers=1)
        self.vae_future = None
        ENC_W_PATH = "zenith_encoder_weights.weights.h5"
        DEC_W_PATH = "zenith_decoder_weights.weights.h5"
        
        if os.path.exists(ENC_W_PATH) and os.path.exists(DEC_W_PATH):
            try:
                self.encoder, self.decoder = self.build_vae_model()
                self.encoder.load_weights(ENC_W_PATH)
                self.decoder.load_weights(DEC_W_PATH)
                self.vae_ok = True
            except Exception as e:
                print(f"VAE Error: {e}")

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

    def vae_quality(self, flat_keypoints):
        z_mean, _, z = self.encoder.predict(flat_keypoints, verbose=0)
        recon = self.decoder.predict(z, verbose=0)
        mse = float(np.mean((flat_keypoints - recon) ** 2))
        q = 100.0 * (1.0 - (mse - 0.0005) / (0.006 - 0.0005))
        return float(np.clip(q, 0, 100)), recon

    def init_mediapipe(self):
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def landmarks_to_flat(self, lms):
        arr = np.array([[lm.x,lm.y,lm.z,lm.visibility] for lm in lms.landmark], dtype=np.float32)
        return arr.flatten()[None,:]

    def majority(self, dq):
        return Counter(dq).most_common(1)[0][0] if dq else None

    def calculate_flow(self, curr_flat, prev_flat, dt, prev_v):
        if prev_flat is None or dt <= 0: return 100.0, 0.0
        diff = curr_flat - prev_flat
        dist = np.linalg.norm(diff)
        velocity = dist / dt
        acceleration = abs(velocity - prev_v) / dt
        punishment = acceleration * 15.0 
        score = 100.0 - punishment
        return float(np.clip(score, 0, 100)), velocity

    def process_dream(self):
        """Generates a hallucinated pose via interpolation."""
        if not self.vae_ok: return None
        
        if self.dream_z_current is None:
            self.dream_z_current = np.random.normal(0, 1, (1, 16))
        if self.dream_z_target is None:
            self.dream_z_target = np.random.normal(0, 1, (1, 16))
            
        z = (1 - self.dream_t) * self.dream_z_current + self.dream_t * self.dream_z_target
        
        self.dream_t += self.DREAM_SPEED
        if self.dream_t >= 1.0:
            self.dream_t = 0.0
            self.dream_z_current = self.dream_z_target
            self.dream_z_target = np.random.normal(0, 1, (1, 16))
            
        recon = self.decoder.predict(z, verbose=0)
        return recon

    # --- CYCLE 21: THE MIRROR (Cinematic Grading) ---
    def apply_cinematic_look(self, img):
        """Applies a 'Neo-Noir' lookup table effect manually."""
        # 1. Contrast Boost
        img = cv2.convertScaleAbs(img, alpha=1.1, beta=0) 
        # 2. Slight Saturation Boost (convert to HSV, bump S, back to BGR)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s = cv2.multiply(s, 1.2) # Boost sat
        s = np.clip(s, 0, 255).astype(hsv.dtype)
        hsv = cv2.merge([h, s, v])
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return img

    def process_frame(self, frame_obj, options):
        # options: {..., use_mirror (implicit or implicit via theme)}
        
        img = frame_obj.to_ndarray(format="bgr24")
        
        # APPLY MIRROR LOOK (Cycle 21)
        img = self.apply_cinematic_look(img)
        
        if options.get('use_tts'):
            self.latest_frame = img.copy()

        t1 = time.time(); dt = t1 - self.t0; self.t0 = t1
        if dt > 0: self.fps = 0.9*self.fps + 0.1*(1.0/dt) if self.fps else (1.0/dt)

        if options.get('use_dream') and self.vae_ok:
            recon = self.process_dream()
            if recon is not None:
                self.ui.draw_ghost(img, recon, alpha=0.8) # Strong ghost in Dream
                cv2.putText(img, "DREAM MODE", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        res = self.pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # Flow
        if options.get('use_flow') and res.pose_landmarks:
            curr_flat = self.landmarks_to_flat(res.pose_landmarks)
            if self.prev_landmarks_array is not None:
                score, velocity = self.calculate_flow(curr_flat, self.prev_landmarks_array, dt, self.prev_velocity)
                alpha_flow = 0.15
                self.current_flow_score = (alpha_flow * score) + ((1-alpha_flow) * self.current_flow_score)
                self.prev_velocity = velocity
            self.prev_landmarks_array = curr_flat
            
            if options.get('use_tts') and (time.time() - self.last_flow_msg_time > self.FLOW_MSG_DEBOUNCE):
                if self.current_flow_score < 40:
                    self.tts_queue.put("Smooth it out. Find your center.")
                    self.last_flow_msg_time = time.time()
                elif self.current_flow_score > 90 and velocity > 0.1:
                    self.tts_queue.put("Excellent flow. Keep moving.")
                    self.last_flow_msg_time = time.time()

        # Debug Draw
        if res.pose_landmarks:
            self.mp_draw.draw_landmarks(img, res.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                                   self.mp_draw.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                   self.mp_draw.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

        label = None
        q_disp = self.last_q

        try:
            if res.pose_landmarks and (self.i % 3 == 0):
                feats = self.landmarks_to_flat(res.pose_landmarks)

                if self.clf_ok and self.pose_classifier:
                    pred = self.pose_classifier.predict(feats)
                    label = self.INT_TO_POSE.get(int(pred[0]), "Unknown")
                    self.hist.append(label)
                    label = self.majority(self.hist)

                    if options.get('use_seq') and self.sequencer and label:
                        seq_status = self.sequencer.update(label, is_stable=False)
                        if seq_status == "Advance" and options.get('use_tts'):
                             self.tts_queue.put(f"Good. Next is {self.sequencer.get_next_goal()}.")

                    if PoseHeuristics and label:
                        lms = {lm_id: [lm.x, lm.y] for lm_id, lm in enumerate(res.pose_landmarks.landmark)}
                        correction = PoseHeuristics.evaluate(label, lms)
                        
                        if correction:
                            self.active_correction = correction
                            self.stability_start_time = None
                            self.is_locked = False
                            
                            text_data = correction['text']
                            spoken_text = text_data
                            if isinstance(text_data, tuple):
                                _, spoken_text = text_data
                            
                            if options.get('use_tts'):
                                now = time.time()
                                if (now - self.last_spoken_time) > self.DEBOUNCE_SECONDS:
                                    self.tts_queue.put(spoken_text)
                                    self.last_spoken_time = now
                        else:
                            self.active_correction = None
                            if options.get('use_gamification'):
                                if self.stability_start_time is None: self.stability_start_time = time.time()
                                duration = time.time() - self.stability_start_time
                                if duration > self.STABILITY_THRESHOLD and not self.is_locked:
                                    self.is_locked = True
                                    self.session.log_stability_event()
                                    
                                    if options.get('use_data') and self.harvester and label:
                                        q_val = self.last_q if self.last_q is not None else 0
                                        self.harvester.save_frame(img, label, q_val)
                                    
                                    if options.get('use_tts'):
                                        self.tts_queue.put(random.choice(self.LOCKED_PHRASES))

                if options.get('use_vae') and self.vae_ok:
                    if (self.i % 6 == 0) and (self.vae_future is None or self.vae_future.done()):
                        self.vae_future = self.vae_pool.submit(self.vae_quality, feats.copy())
                    if self.vae_future and self.vae_future.done():
                        q, recon = self.vae_future.result()
                        alpha = 0.25
                        self.last_q = q if self.last_q is None else (alpha*q + (1-alpha)*self.last_q)
                        q_disp = self.last_q
                        self.last_recon = recon

                if label or q_disp is not None:
                    self.last_label = label or self.last_label
                    self.last_ok_ts = time.time()

        except Exception as e:
            print(f"Core Error: {e}")

        self.i += 1
        if (time.time() - self.last_ok_ts) < self.TTL: label = label or self.last_label

        is_stable_now = (self.stability_start_time is not None)
        self.session.update(label, self.current_flow_score if options.get('use_flow') else None, is_stable_now, self.fps)

        # --- DRAWING ---
        if self.ui:
            if options.get('use_grid'): self.ui.draw_grid(img)
            
            # SMART GHOST LOGIC (Cycle 21)
            # If Q > 85 (Good form) -> Alpha 0.1 (Faint)
            # If Q < 60 (Bad form) -> Alpha 0.5 (Strong)
            ghost_alpha = 0.3 # Default
            if q_disp is not None:
                if q_disp > 85: ghost_alpha = 0.1
                elif q_disp < 60: ghost_alpha = 0.5
            
            if options.get('use_ghost') and self.last_recon is not None:
                self.ui.draw_ghost(img, self.last_recon, alpha=ghost_alpha)

            if options.get('use_gamification') and self.stability_start_time is not None and res.pose_landmarks:
                duration = time.time() - self.stability_start_time
                progress = min(duration / self.STABILITY_THRESHOLD, 1.0)
                self.ui.draw_halo(img, res.pose_landmarks, progress, self.is_locked)

            if options.get('use_ar') and self.active_correction:
                 if 'vector' in self.active_correction:
                    start_pt, end_pt = self.active_correction['vector']
                    ar_text = self.active_correction.get('hud_text', self.active_correction.get('text'))
                    if isinstance(ar_text, tuple): ar_text = ar_text[0]
                    self.ui.draw_arrow(img, start_pt, end_pt, text=ar_text)

            if options.get('use_flow'):
                self.ui.draw_flow_bar(img, self.current_flow_score)

            if options.get('use_seq') and self.sequencer:
                self.ui.draw_sequencer(img, self.sequencer.get_current_goal(), self.sequencer.get_next_goal(), self.sequencer.get_progress())

            self.ui.draw_hud(img, label, q_disp, self.fps)
            
            if options.get('use_data') and self.harvester:
                 count = self.harvester.get_stats()
                 cv2.putText(img, f"DATA: {count}", (10, img.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return av.VideoFrame.from_ndarray(img, format="bgr24")
