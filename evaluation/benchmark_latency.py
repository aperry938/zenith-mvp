"""
ZENith Real-Time Feasibility Benchmark.

Measures end-to-end latency of the biomechanical feature pipeline
to confirm real-time operation at 30fps (33.3ms budget).

Components timed:
1. Landmark extraction (MediaPipe) — N/A here (measured separately)
2. Raw feature flattening (132-dim)
3. Biomechanical feature extraction (30-dim)
4. Pose classification (RF, XGB, MLP)
5. Quality score computation
6. Full pipeline: features + classify + quality
"""

import os
import sys
import time
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from biomechanical_features import (
    extract_biomechanical_features, compute_pose_quality_score,
    get_deviations, StabilityTracker, FILE_PREFIX_TO_PROFILE
)


def benchmark(func, n_iters=1000, warmup=50):
    """Benchmark a function, return timing stats in ms."""
    # Warmup
    for _ in range(warmup):
        func()

    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        func()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    return {
        "mean_ms": round(np.mean(times), 4),
        "std_ms": round(np.std(times), 4),
        "p50_ms": round(np.percentile(times, 50), 4),
        "p95_ms": round(np.percentile(times, 95), 4),
        "p99_ms": round(np.percentile(times, 99), 4),
        "min_ms": round(np.min(times), 4),
        "max_ms": round(np.max(times), 4),
        "fps_achievable": round(1000.0 / np.mean(times), 1),
    }


def main():
    keypoints_dir = os.path.join(os.path.dirname(__file__), '..', 'ZENith_Data', 'keypoints')
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)

    # Load a sample frame
    sample_file = os.path.join(keypoints_dir, 'Warrior2_Correct_1.npy')
    data = np.load(sample_file)
    frame = data[data.shape[0] // 2]  # (33, 4) single frame
    flat = frame.reshape(1, -1).astype(np.float32)  # (1, 132)

    tracker = StabilityTracker(buffer_size=15)

    print("ZENith Latency Benchmark")
    print("=" * 60)
    print(f"Budget: 33.3ms per frame (30fps)")
    print()

    results = {}

    # 1. Raw flattening
    print("[1] Raw feature flattening (132-dim)...")
    r = benchmark(lambda: frame.reshape(1, -1).astype(np.float32))
    results["raw_flatten"] = r
    print(f"  Mean: {r['mean_ms']:.4f}ms | P95: {r['p95_ms']:.4f}ms | Max FPS: {r['fps_achievable']}")

    # 2. Biomechanical extraction
    print("[2] Biomechanical feature extraction (30-dim)...")
    r = benchmark(lambda: extract_biomechanical_features(frame, tracker))
    results["bio_extraction"] = r
    print(f"  Mean: {r['mean_ms']:.4f}ms | P95: {r['p95_ms']:.4f}ms | Max FPS: {r['fps_achievable']}")

    # 3. Quality score
    print("[3] Quality score computation...")
    bio_feats = extract_biomechanical_features(frame, tracker)
    r = benchmark(lambda: compute_pose_quality_score(bio_feats, "Warrior II"))
    results["quality_score"] = r
    print(f"  Mean: {r['mean_ms']:.4f}ms | P95: {r['p95_ms']:.4f}ms | Max FPS: {r['fps_achievable']}")

    # 4. Deviation detection
    print("[4] Deviation detection...")
    r = benchmark(lambda: get_deviations(bio_feats, "Warrior II"))
    results["deviation_detection"] = r
    print(f"  Mean: {r['mean_ms']:.4f}ms | P95: {r['p95_ms']:.4f}ms | Max FPS: {r['fps_achievable']}")

    # 5. Classifier inference (if models available)
    try:
        import joblib
        clf_path = os.path.join(os.path.dirname(__file__), '..', 'zenith_pose_classifier.pkl')
        if os.path.exists(clf_path):
            clf = joblib.load(clf_path)

            # RF on raw
            print("[5a] RF classification (raw 132-dim)...")
            r = benchmark(lambda: clf.predict(flat), n_iters=500)
            results["rf_raw"] = r
            print(f"  Mean: {r['mean_ms']:.4f}ms | P95: {r['p95_ms']:.4f}ms | Max FPS: {r['fps_achievable']}")

            # RF on bio
            bio_flat = bio_feats[None, :].astype(np.float32)
            # Can't predict with existing model (wrong dimensions), note this
            print("  (Note: existing RF trained on 132-dim; bio requires retraining)")
    except Exception as e:
        print(f"  Classifier benchmark skipped: {e}")

    # 6. Full pipeline
    print("[6] Full pipeline: bio extract + quality + deviations...")
    def full_pipeline():
        bf = extract_biomechanical_features(frame, tracker)
        q = compute_pose_quality_score(bf, "Warrior II")
        d = get_deviations(bf, "Warrior II")
        return bf, q, d

    r = benchmark(full_pipeline, n_iters=500)
    results["full_pipeline"] = r
    print(f"  Mean: {r['mean_ms']:.4f}ms | P95: {r['p95_ms']:.4f}ms | Max FPS: {r['fps_achievable']}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total = results["full_pipeline"]["mean_ms"]
    budget = 33.3
    print(f"Full pipeline latency: {total:.2f}ms ({total/budget*100:.1f}% of 30fps budget)")
    print(f"Achievable FPS: {results['full_pipeline']['fps_achievable']}")
    print(f"{'PASS' if total < budget else 'FAIL'}: {'Within' if total < budget else 'Exceeds'} real-time budget")

    # Save
    out_path = os.path.join(results_dir, 'latency_benchmark.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
