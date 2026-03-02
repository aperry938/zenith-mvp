"""
Biomechanical VAE: Deeper architecture on biomechanical features.

Replaces the original VAE (132→64→16 latent) with:
- Standard VAE: 30→64→32→8 latent on biomechanical features
- Pose-Conditioned VAE (c-VAE): concatenates one-hot pose label

The biomechanical input is more compact and meaningful than raw landmarks,
enabling a deeper architecture with a smaller latent space that better
captures the structure of correct pose form.

Training strategy:
- Train on correct-form videos only (quality > 85)
- Reconstruction error serves as quality proxy
- Low error = close to correct-form manifold
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Suppress TF warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

sys.path.insert(0, os.path.dirname(__file__))
from biomechanical_features import (
    extract_biomechanical_batch, StabilityTracker,
    FILE_PREFIX_TO_PROFILE, NUM_FEATURES
)


# ── Sampling Layer ──────────────────────────────────────────────────────────

class Sampling(layers.Layer):
    """Reparameterization trick: z = z_mean + eps * exp(0.5 * z_log_var)"""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        eps = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * eps


# ── Standard Biomechanical VAE ──────────────────────────────────────────────

def build_bio_vae(input_dim=30, latent_dim=8):
    """
    Build deeper VAE on biomechanical features.

    Architecture:
        Encoder: 30 → 64 (relu) → 32 (relu) → z_mean(8), z_log_var(8)
        Decoder: 8 → 32 (relu) → 64 (relu) → 30 (sigmoid)
    """
    # Encoder
    enc_in = keras.Input(shape=(input_dim,), name="encoder_input")
    x = layers.Dense(64, activation="relu", name="enc_dense1")(enc_in)
    x = layers.Dense(32, activation="relu", name="enc_dense2")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(enc_in, [z_mean, z_log_var, z], name="bio_encoder")

    # Decoder
    dec_in = keras.Input(shape=(latent_dim,), name="decoder_input")
    x = layers.Dense(32, activation="relu", name="dec_dense1")(dec_in)
    x = layers.Dense(64, activation="relu", name="dec_dense2")(x)
    dec_out = layers.Dense(input_dim, activation="sigmoid", name="dec_output")(x)
    decoder = keras.Model(dec_in, dec_out, name="bio_decoder")

    return encoder, decoder


# ── Pose-Conditioned VAE (c-VAE) ───────────────────────────────────────────

def build_conditional_vae(input_dim=30, n_poses=10, latent_dim=8):
    """
    Build conditional VAE that conditions on pose class.

    Concatenates one-hot pose label at both encoder and decoder input.
    Learns per-pose quality manifolds within a shared latent space.

    Architecture:
        Encoder: (30 + 10) → 64 → 32 → z_mean(8), z_log_var(8)
        Decoder: (8 + 10) → 32 → 64 → 30
    """
    # Encoder
    feat_in = keras.Input(shape=(input_dim,), name="features")
    pose_in = keras.Input(shape=(n_poses,), name="pose_label")
    enc_concat = layers.Concatenate()([feat_in, pose_in])
    x = layers.Dense(64, activation="relu", name="enc_dense1")(enc_concat)
    x = layers.Dense(32, activation="relu", name="enc_dense2")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model([feat_in, pose_in], [z_mean, z_log_var, z], name="cvae_encoder")

    # Decoder
    z_in = keras.Input(shape=(latent_dim,), name="latent_input")
    pose_dec_in = keras.Input(shape=(n_poses,), name="pose_label_dec")
    dec_concat = layers.Concatenate()([z_in, pose_dec_in])
    x = layers.Dense(32, activation="relu", name="dec_dense1")(dec_concat)
    x = layers.Dense(64, activation="relu", name="dec_dense2")(x)
    dec_out = layers.Dense(input_dim, activation="sigmoid", name="dec_output")(x)
    decoder = keras.Model([z_in, pose_dec_in], dec_out, name="cvae_decoder")

    return encoder, decoder


# ── VAE Training ────────────────────────────────────────────────────────────

class VAETrainer:
    """Handles VAE training with custom loss (reconstruction + KL divergence)."""

    def __init__(self, encoder, decoder, kl_weight=0.001):
        self.encoder = encoder
        self.decoder = decoder
        self.kl_weight = kl_weight
        self.optimizer = keras.optimizers.Adam(learning_rate=1e-3)
        self.history = {"loss": [], "recon_loss": [], "kl_loss": []}

    @tf.function
    def train_step(self, x, pose_onehot=None):
        """Single training step."""
        with tf.GradientTape() as tape:
            # Encode
            if pose_onehot is not None:
                z_mean, z_log_var, z = self.encoder([x, pose_onehot])
            else:
                z_mean, z_log_var, z = self.encoder(x)

            # Decode
            if pose_onehot is not None:
                recon = self.decoder([z, pose_onehot])
            else:
                recon = self.decoder(z)

            # Reconstruction loss (MSE)
            recon_loss = tf.reduce_mean(tf.reduce_sum(tf.square(x - recon), axis=1))

            # KL divergence
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
            )

            total_loss = recon_loss + self.kl_weight * kl_loss

        # Update weights
        trainable_vars = self.encoder.trainable_variables + self.decoder.trainable_variables
        grads = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(grads, trainable_vars))

        return total_loss, recon_loss, kl_loss

    def train(self, X, pose_labels=None, epochs=100, batch_size=32, verbose=True):
        """Train the VAE."""
        n = len(X)
        n_batches = max(1, n // batch_size)

        for epoch in range(epochs):
            # Shuffle
            perm = np.random.permutation(n)
            X_shuffled = X[perm]
            pose_shuffled = pose_labels[perm] if pose_labels is not None else None

            epoch_loss = 0
            epoch_recon = 0
            epoch_kl = 0

            for i in range(n_batches):
                start = i * batch_size
                end = min(start + batch_size, n)
                batch_x = tf.cast(X_shuffled[start:end], tf.float32)
                batch_pose = tf.cast(pose_shuffled[start:end], tf.float32) if pose_shuffled is not None else None

                loss, recon, kl = self.train_step(batch_x, batch_pose)
                epoch_loss += float(loss)
                epoch_recon += float(recon)
                epoch_kl += float(kl)

            epoch_loss /= n_batches
            epoch_recon /= n_batches
            epoch_kl /= n_batches

            self.history["loss"].append(epoch_loss)
            self.history["recon_loss"].append(epoch_recon)
            self.history["kl_loss"].append(epoch_kl)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1:3d}/{epochs}: loss={epoch_loss:.4f} "
                      f"recon={epoch_recon:.4f} kl={epoch_kl:.4f}")


# ── Data Loading ────────────────────────────────────────────────────────────

def load_training_data(keypoints_dir, correct_only=True):
    """
    Load biomechanical features for VAE training.

    Args:
        correct_only: If True, only load correct-form videos (for quality manifold)

    Returns:
        features: (N, 30) array of biomechanical features
        pose_indices: (N,) array of pose class indices
        pose_names: list of pose name strings (ordered by index)
    """
    all_features = []
    all_pose_indices = []

    pose_names = sorted(set(FILE_PREFIX_TO_PROFILE.values()))
    pose_to_idx = {name: i for i, name in enumerate(pose_names)}

    files = sorted(f for f in os.listdir(keypoints_dir) if f.endswith('.npy'))

    for fname in files:
        if correct_only and "Correct" not in fname:
            continue

        prefix = None
        for known_prefix in FILE_PREFIX_TO_PROFILE:
            if fname.startswith(known_prefix + "_"):
                prefix = known_prefix
                break
        if prefix is None:
            continue

        pose_label = FILE_PREFIX_TO_PROFILE[prefix]
        pose_idx = pose_to_idx[pose_label]

        data = np.load(os.path.join(keypoints_dir, fname))
        n = data.shape[0]

        # Extract biomechanical features
        tracker = StabilityTracker()
        bio = extract_biomechanical_batch(data, tracker)

        # Use held-pose region (middle 50%) sampled every 3rd frame
        start = int(n * 0.25)
        end = int(n * 0.75)
        for idx in range(start, end, 3):
            all_features.append(bio[idx])
            all_pose_indices.append(pose_idx)

    features = np.array(all_features, dtype=np.float32)
    pose_indices = np.array(all_pose_indices, dtype=np.int32)

    return features, pose_indices, pose_names


# ── Main Training Script ───────────────────────────────────────────────────

def main():
    keypoints_dir = os.path.join(os.path.dirname(__file__), 'ZENith_Data', 'keypoints')
    models_dir = os.path.join(os.path.dirname(__file__), 'ZENith_Data', 'models')
    os.makedirs(models_dir, exist_ok=True)

    print("ZENith Biomechanical VAE Training")
    print("=" * 60)

    # 1. Load data
    print("\n[1/4] Loading training data (correct-form only)...")
    features, pose_indices, pose_names = load_training_data(keypoints_dir, correct_only=True)
    print(f"  Samples: {len(features)}, Features: {features.shape[1]}, Poses: {len(pose_names)}")

    n_poses = len(pose_names)
    pose_onehot = np.eye(n_poses, dtype=np.float32)[pose_indices]

    # 2. Train standard Bio-VAE
    print("\n[2/4] Training Standard Bio-VAE (30→64→32→8)...")
    encoder, decoder = build_bio_vae(input_dim=NUM_FEATURES, latent_dim=8)
    trainer = VAETrainer(encoder, decoder, kl_weight=0.001)
    trainer.train(features, epochs=100, batch_size=32)

    # Save weights
    encoder.save_weights(os.path.join(models_dir, 'bio_encoder.weights.h5'))
    decoder.save_weights(os.path.join(models_dir, 'bio_decoder.weights.h5'))
    print("  Saved: bio_encoder.weights.h5, bio_decoder.weights.h5")

    # 3. Train conditional VAE
    print("\n[3/4] Training Conditional Bio-VAE (c-VAE)...")
    c_encoder, c_decoder = build_conditional_vae(
        input_dim=NUM_FEATURES, n_poses=n_poses, latent_dim=8
    )
    c_trainer = VAETrainer(c_encoder, c_decoder, kl_weight=0.001)
    c_trainer.train(features, pose_labels=pose_onehot, epochs=100, batch_size=32)

    # Save weights
    c_encoder.save_weights(os.path.join(models_dir, 'cvae_encoder.weights.h5'))
    c_decoder.save_weights(os.path.join(models_dir, 'cvae_decoder.weights.h5'))
    print("  Saved: cvae_encoder.weights.h5, cvae_decoder.weights.h5")

    # 4. Evaluate reconstruction quality
    print("\n[4/4] Evaluating reconstruction quality...")

    # Load ALL data (correct + incorrect) for evaluation
    all_features, all_pose_indices, _ = load_training_data(keypoints_dir, correct_only=False)
    all_onehot = np.eye(n_poses, dtype=np.float32)[all_pose_indices]

    # Standard VAE reconstruction
    z_mean, _, z = encoder.predict(all_features, verbose=0)
    recon = decoder.predict(z, verbose=0)
    mse_standard = np.mean((all_features - recon) ** 2, axis=1)

    # c-VAE reconstruction
    cz_mean, _, cz = c_encoder.predict([all_features, all_onehot], verbose=0)
    crecon = c_decoder.predict([cz, all_onehot], verbose=0)
    mse_cvae = np.mean((all_features - crecon) ** 2, axis=1)

    # Separate correct vs incorrect
    # Reload to get correct/incorrect labels
    correct_mask = []
    files = sorted(f for f in os.listdir(keypoints_dir) if f.endswith('.npy'))
    sample_idx = 0
    for fname in files:
        prefix = None
        for known_prefix in FILE_PREFIX_TO_PROFILE:
            if fname.startswith(known_prefix + "_"):
                prefix = known_prefix
                break
        if prefix is None:
            continue

        data = np.load(os.path.join(keypoints_dir, fname))
        n = data.shape[0]
        start = int(n * 0.25)
        end = int(n * 0.75)
        n_samples = len(range(start, end, 3))
        is_correct = "Correct" in fname
        correct_mask.extend([is_correct] * n_samples)

    correct_mask = np.array(correct_mask[:len(all_features)])

    print(f"\n{'Model':15s} {'Correct MSE':>15s} {'Incorrect MSE':>15s} {'Ratio':>8s}")
    print("-" * 55)

    for name, mse_arr in [("Standard VAE", mse_standard), ("c-VAE", mse_cvae)]:
        c_mse = np.mean(mse_arr[correct_mask])
        i_mse = np.mean(mse_arr[~correct_mask])
        ratio = i_mse / (c_mse + 1e-8)
        print(f"{name:15s} {c_mse:>13.6f}   {i_mse:>13.6f}   {ratio:>6.2f}×")

    # Save training history
    history = {
        "standard_vae": trainer.history,
        "cvae": c_trainer.history,
        "evaluation": {
            "standard_correct_mse": float(np.mean(mse_standard[correct_mask])),
            "standard_incorrect_mse": float(np.mean(mse_standard[~correct_mask])),
            "cvae_correct_mse": float(np.mean(mse_cvae[correct_mask])),
            "cvae_incorrect_mse": float(np.mean(mse_cvae[~correct_mask])),
        }
    }

    with open(os.path.join(models_dir, 'vae_training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
