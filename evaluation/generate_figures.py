"""
Generate publication-ready figures for ZENith evaluation.

Figures:
1. Ablation bar chart (accuracy × condition × classifier)
2. Confusion matrices (best configuration)
3. Quality score scatter plot (bio quality vs annotation rating)
4. Per-pose quality discrimination box plot
5. t-SNE visualization of feature spaces (RAW vs BIO)
6. Feature importance plot (Random Forest, BIO condition)
"""

import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

# Publication style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

COLORS = {
    'RAW': '#4C72B0',
    'BIO': '#DD8452',
    'RAW+BIO': '#55A868',
    'BIO+TEMPORAL': '#C44E52',
}


def plot_ablation_bars(results_path, output_dir):
    """Figure 1: Ablation study bar chart."""
    with open(results_path) as f:
        results = json.load(f)

    conditions = list(results.keys())
    classifiers = list(results[conditions[0]]["classifiers"].keys())

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax_idx, metric in enumerate(['accuracy_mean', 'f1_macro_mean']):
        ax = axes[ax_idx]
        x = np.arange(len(classifiers))
        width = 0.18
        offsets = np.linspace(-width * 1.5, width * 1.5, len(conditions))

        for i, cond in enumerate(conditions):
            values = []
            errors = []
            for clf_name in classifiers:
                r = results[cond]["classifiers"][clf_name]
                values.append(r[metric])
                ci = r[metric.replace('_mean', '_ci95')]
                error = (r[metric] - ci[0] + ci[1] - r[metric]) / 2
                errors.append(error)

            bars = ax.bar(x + offsets[i], values, width, label=cond,
                         color=COLORS[cond], yerr=errors, capsize=3,
                         error_kw={'linewidth': 0.8})

        ax.set_xlabel('Classifier')
        ax.set_ylabel('Accuracy' if 'accuracy' in metric else 'F1 Macro')
        ax.set_title('Classification Accuracy' if 'accuracy' in metric else 'F1 Score (Macro)')
        ax.set_xticks(x)
        ax.set_xticklabels(classifiers)
        ax.legend(loc='lower right', framealpha=0.9)
        ax.set_ylim(0.5, 1.05)
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'fig_ablation_results.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_confusion_matrix(results_path, output_dir):
    """Figure 2: Confusion matrix for best configuration."""
    with open(results_path) as f:
        results = json.load(f)

    # Find best configuration
    best_acc = 0
    best_result = None
    for cond, cr in results.items():
        for clf_name, r in cr["classifiers"].items():
            if r["accuracy_mean"] > best_acc:
                best_acc = r["accuracy_mean"]
                best_result = r
                best_cond = cond
                best_clf = clf_name

    if best_result is None:
        return

    cm = np.array(best_result["confusion_matrix"])
    class_names = best_result["class_names"]
    n = len(class_names)

    # Normalize by row (true labels)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)

    # Add text annotations
    for i in range(n):
        for j in range(n):
            val = cm_norm[i, j]
            count = cm[i, j]
            color = 'white' if val > 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}\n({count})',
                   ha='center', va='center', color=color, fontsize=7)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Confusion Matrix: {best_cond} × {best_clf}\n(Accuracy: {best_acc:.4f})')

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    path = os.path.join(output_dir, 'fig_confusion_matrix.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_quality_discrimination(quality_path, output_dir):
    """Figure 3: Quality score discrimination box plot."""
    with open(quality_path) as f:
        quality = json.load(f)

    per_pose = quality["per_pose"]
    poses = sorted(per_pose.keys(), key=lambda p: -per_pose[p]["delta"])

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(poses))
    width = 0.35

    correct_means = [per_pose[p]["correct_mean"] for p in poses]
    incorrect_means = [per_pose[p]["incorrect_mean"] for p in poses]
    correct_stds = [per_pose[p]["correct_std"] for p in poses]
    incorrect_stds = [per_pose[p]["incorrect_std"] for p in poses]

    bars1 = ax.bar(x - width/2, correct_means, width, yerr=correct_stds,
                   label='Correct Form', color='#55A868', capsize=3,
                   error_kw={'linewidth': 0.8})
    bars2 = ax.bar(x + width/2, incorrect_means, width, yerr=incorrect_stds,
                   label='Incorrect Form', color='#C44E52', capsize=3,
                   error_kw={'linewidth': 0.8})

    # Add delta annotations
    for i, pose in enumerate(poses):
        delta = per_pose[pose]["delta"]
        y_max = max(correct_means[i] + correct_stds[i],
                    incorrect_means[i] + incorrect_stds[i])
        if delta > 0:
            ax.annotate(f'+{delta:.0f}', xy=(i, y_max + 3),
                       ha='center', fontsize=7, color='#2C7BB6', fontweight='bold')

    ax.set_xlabel('Yoga Pose')
    ax.set_ylabel('Biomechanical Quality Score')
    ax.set_title('Quality Score Discrimination: Correct vs. Incorrect Form')
    ax.set_xticks(x)
    ax.set_xticklabels(poses, rotation=30, ha='right', fontsize=8)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 115)

    plt.tight_layout()
    path = os.path.join(output_dir, 'fig_quality_discrimination.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_quality_scatter(quality_path, output_dir):
    """Figure 4: Quality score vs annotation rating scatter."""
    with open(quality_path) as f:
        quality = json.load(f)

    results = quality["individual_results"]
    scores = [r["bio_quality_mean"] for r in results]
    ratings = [r["annotation_rating"] for r in results]
    gt_labels = [r["ground_truth"] for r in results]

    fig, ax = plt.subplots(figsize=(6, 5))

    for gt, color, marker in [("correct", "#55A868", "o"), ("incorrect", "#C44E52", "s")]:
        mask = [g == gt for g in gt_labels]
        s = [scores[i] for i in range(len(scores)) if mask[i]]
        r = [ratings[i] for i in range(len(ratings)) if mask[i]]
        # Add jitter to rating for visibility
        r_jittered = [ri + np.random.uniform(-0.15, 0.15) for ri in r]
        ax.scatter(r_jittered, s, c=color, marker=marker, alpha=0.6,
                  s=40, label=gt.capitalize(), edgecolors='white', linewidth=0.5)

    correlation = quality["rating_correlation"]
    ax.set_xlabel('Expert Quality Rating (1-5)')
    ax.set_ylabel('Biomechanical Quality Score')
    ax.set_title(f'Quality Score vs. Rating (Spearman ρ={correlation["spearman_rho"]:.3f})')
    ax.legend()
    ax.set_xlim(0.5, 5.5)
    ax.set_ylim(-5, 110)

    plt.tight_layout()
    path = os.path.join(output_dir, 'fig_quality_scatter.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_tsne(output_dir):
    """Figure 5: t-SNE of RAW vs BIO feature spaces."""
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("  Skipping t-SNE (sklearn not available)")
        return

    keypoints_dir = os.path.join(os.path.dirname(__file__), '..', 'ZENith_Data', 'keypoints')

    from biomechanical_features import (
        extract_biomechanical_batch, StabilityTracker, FILE_PREFIX_TO_PROFILE
    )

    # Load subset of data
    raw_list, bio_list, label_list = [], [], []
    files = sorted(f for f in os.listdir(keypoints_dir) if f.endswith('.npy'))

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
        mid = n // 2

        # Take 3 frames from held region
        for idx in [int(n*0.3), mid, int(n*0.7)]:
            raw_list.append(data[idx].reshape(-1))
            bio_list.append(None)  # Placeholder
            label_list.append(FILE_PREFIX_TO_PROFILE[prefix])

    # Compute bio features
    for i, fname in enumerate(files):
        prefix = None
        for known_prefix in FILE_PREFIX_TO_PROFILE:
            if fname.startswith(known_prefix + "_"):
                prefix = known_prefix
                break
        if prefix is None:
            continue

        data = np.load(os.path.join(keypoints_dir, fname))
        n = data.shape[0]
        tracker = StabilityTracker()
        bio = extract_biomechanical_batch(data, tracker)
        for j, idx in enumerate([int(n*0.3), n//2, int(n*0.7)]):
            bio_idx = i * 3 + j
            if bio_idx < len(bio_list):
                bio_list[bio_idx] = bio[idx]

    # Filter out None entries
    valid = [i for i in range(len(bio_list)) if bio_list[i] is not None]
    raw_arr = np.array([raw_list[i] for i in valid])
    bio_arr = np.array([bio_list[i] for i in valid])
    labels = [label_list[i] for i in valid]

    # Unique labels and colors
    unique_labels = sorted(set(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    label_to_color = {l: colors[i] for i, l in enumerate(unique_labels)}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (feat_arr, title) in zip(axes, [(raw_arr, 'RAW (132-dim)'), (bio_arr, 'BIO (30-dim)')]):
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
        embedding = tsne.fit_transform(feat_arr)

        for label in unique_labels:
            mask = [l == label for l in labels]
            ax.scatter(embedding[mask, 0], embedding[mask, 1],
                      c=[label_to_color[label]], label=label, s=15, alpha=0.7)

        ax.set_title(f't-SNE: {title}')
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')

    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
    plt.tight_layout()
    path = os.path.join(output_dir, 'fig_tsne_comparison.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_latency(results_path, output_dir):
    """Figure 6: Latency comparison."""
    with open(results_path) as f:
        results = json.load(f)

    conditions = list(results.keys())
    classifiers = list(results[conditions[0]]["classifiers"].keys())

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(classifiers))
    width = 0.18
    offsets = np.linspace(-width * 1.5, width * 1.5, len(conditions))

    for i, cond in enumerate(conditions):
        latencies = []
        for clf_name in classifiers:
            r = results[cond]["classifiers"][clf_name]
            latencies.append(r["latency"]["mean_ms"])

        ax.bar(x + offsets[i], latencies, width, label=cond, color=COLORS[cond])

    ax.set_xlabel('Classifier')
    ax.set_ylabel('Inference Latency (ms)')
    ax.set_title('Single-Sample Inference Latency')
    ax.set_xticks(x)
    ax.set_xticklabels(classifiers)
    ax.legend()
    ax.axhline(y=33.3, color='red', linestyle='--', alpha=0.5, label='30fps budget')
    ax.set_ylim(0, None)

    plt.tight_layout()
    path = os.path.join(output_dir, 'fig_latency.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def main():
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    figures_dir = os.path.join(results_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    ablation_path = os.path.join(results_dir, 'ablation_results.json')
    quality_path = os.path.join(results_dir, 'quality_validation.json')

    print("Generating publication-ready figures...")
    print("=" * 50)

    if os.path.exists(ablation_path):
        print("\n[1] Ablation bar chart...")
        plot_ablation_bars(ablation_path, figures_dir)

        print("\n[2] Confusion matrix...")
        plot_confusion_matrix(ablation_path, figures_dir)

        print("\n[6] Latency comparison...")
        plot_latency(ablation_path, figures_dir)
    else:
        print("  Skipping ablation figures (run train_and_evaluate.py first)")

    if os.path.exists(quality_path):
        print("\n[3] Quality discrimination box plot...")
        plot_quality_discrimination(quality_path, figures_dir)

        print("\n[4] Quality scatter plot...")
        plot_quality_scatter(quality_path, figures_dir)
    else:
        print("  Skipping quality figures (run quality_validation.py first)")

    print("\n[5] t-SNE comparison...")
    plot_tsne(figures_dir)

    print("\nAll figures generated.")


if __name__ == "__main__":
    main()
