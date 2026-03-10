import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from models.fusion import MultiModalFramework

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

PROFILE_NAMES  = ['High Achiever', 'Struggling', 'Social Learner', 'Quiet Worker']
PROFILE_COLORS = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12']


# ── Load & Normalize ───────────────────────────────────────────
def load_data():
    academic   = pd.read_csv('data/academic.csv').values
    behavioral = pd.read_csv('data/behavioral.csv').values
    activity   = pd.read_csv('data/activity.csv').values

    normalized = []
    for d in [academic, behavioral, activity]:
        sc = StandardScaler()
        normalized.append(sc.fit_transform(d))

    return [torch.FloatTensor(d) for d in normalized]


# ── Extract Embeddings ─────────────────────────────────────────
def extract_embeddings(model, data):
    model.eval()
    a, b, c = [x.to(device) for x in data]
    with torch.no_grad():
        unified, attn = model(a, b, c)
    return unified.cpu().numpy(), attn.cpu().numpy()


# ── Plot 1: UMAP ───────────────────────────────────────────────
def plot_umap(embeddings, true_labels, pred_labels):
    try:
        from umap import UMAP
        reducer = UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.1)
    except ImportError:
        from sklearn.manifold import TSNE
        print("UMAP not found, using t-SNE instead")
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)

    print("Computing 2D projection (this takes ~30 seconds)...")
    reduced = reducer.fit_transform(embeddings)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Multi-Modal Representation Learning — Embedding Space', 
                 fontsize=14, fontweight='bold')

    # Left: True labels (ground truth)
    ax = axes[0]
    for i, (name, color) in enumerate(zip(PROFILE_NAMES, PROFILE_COLORS)):
        mask = true_labels == i
        ax.scatter(reduced[mask, 0], reduced[mask, 1],
                   c=color, label=name, alpha=0.6, s=15, edgecolors='none')
    ax.set_title('Ground Truth Profiles\n(model never saw these labels)', fontsize=11)
    ax.legend(markerscale=2, fontsize=9)
    ax.set_xlabel('UMAP-1'); ax.set_ylabel('UMAP-2')
    ax.grid(True, alpha=0.2)

    # Right: Discovered clusters (unsupervised)
    ax = axes[1]
    cluster_colors = ['#8e44ad', '#16a085', '#d35400', '#2980b9']
    for i in range(4):
        mask = pred_labels == i
        ax.scatter(reduced[mask, 0], reduced[mask, 1],
                   c=cluster_colors[i], label=f'Cluster {i}',
                   alpha=0.6, s=15, edgecolors='none')
    ax.set_title('Discovered Clusters\n(KMeans on learned embeddings)', fontsize=11)
    ax.legend(markerscale=2, fontsize=9)
    ax.set_xlabel('UMAP-1'); ax.set_ylabel('UMAP-2')
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig('outputs/umap_embeddings.png', dpi=150, bbox_inches='tight')
    print("✅ Saved: outputs/umap_embeddings.png")
    plt.show()
    return reduced


# ── Plot 2: Attention Weights ──────────────────────────────────
def plot_attention(attn_weights, true_labels):
    MODALITIES = ['Academic', 'Behavioral', 'Activity']

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Modality Attention Weights — Explainability', 
                 fontsize=14, fontweight='bold')

    # Left: Average attention per profile
    ax = axes[0]
    x   = np.arange(len(MODALITIES))
    w   = 0.18
    for i, (name, color) in enumerate(zip(PROFILE_NAMES, PROFILE_COLORS)):
        avg = attn_weights[true_labels == i].mean(axis=0)
        ax.bar(x + i * w, avg, width=w, label=name, color=color, alpha=0.85)

    ax.set_xticks(x + w * 1.5)
    ax.set_xticklabels(MODALITIES, fontsize=11)
    ax.set_ylabel('Average Attention Weight')
    ax.set_title('Which modality each profile relies on')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis='y')

    # Right: Overall average as pie chart
    ax = axes[1]
    overall_avg = attn_weights.mean(axis=0)
    wedges, texts, autotexts = ax.pie(
        overall_avg,
        labels=MODALITIES,
        colors=['#3498db', '#e67e22', '#2ecc71'],
        autopct='%1.1f%%',
        startangle=90,
        textprops={'fontsize': 12}
    )
    ax.set_title('Overall modality contribution\nacross all 5000 students')

    plt.tight_layout()
    plt.savefig('outputs/attention_weights.png', dpi=150, bbox_inches='tight')
    print("✅ Saved: outputs/attention_weights.png")
    plt.show()


# ── Plot 3: Loss Curve ─────────────────────────────────────────
def plot_loss():
    loss = np.load('outputs/loss_history.npy')
    plt.figure(figsize=(10, 5))
    plt.plot(loss, color='#3498db', linewidth=1.5, label='Training Loss')
    plt.axhline(y=min(loss), color='#e74c3c', linestyle='--', 
                linewidth=1, label=f'Best: {min(loss):.4f}')
    plt.fill_between(range(len(loss)), loss, alpha=0.1, color='#3498db')
    plt.xlabel('Epoch'); plt.ylabel('NT-Xent Loss')
    plt.title('Contrastive Training Loss (SimCLR)')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('outputs/loss_curve.png', dpi=150)
    print("✅ Saved: outputs/loss_curve.png")
    plt.show()


# ── Main ───────────────────────────────────────────────────────
def main():
    print("Loading model...")
    model = MultiModalFramework(embedding_dim=64, unified_dim=128).to(device)
    checkpoint = torch.load('models/best_model.pt', map_location=device)
    model.load_state_dict(checkpoint['model'])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} "
          f"(loss={checkpoint['loss']:.4f})")

    print("Loading data...")
    data        = load_data()
    true_labels = np.load('data/labels.npy')

    print("Extracting embeddings...")
    embeddings, attn_weights = extract_embeddings(model, data)
    print(f"Embeddings shape: {embeddings.shape}")

    # KMeans clustering on learned embeddings
    print("Clustering embeddings...")
    kmeans      = KMeans(n_clusters=4, random_state=42, n_init=10)
    pred_labels = kmeans.fit_predict(embeddings)

    # Metrics
    sil_score = silhouette_score(embeddings, pred_labels)
    ari_score = adjusted_rand_score(true_labels, pred_labels)
    print(f"\n{'='*45}")
    print(f"  Silhouette Score : {sil_score:.4f}  (higher=better, max=1.0)")
    print(f"  Adjusted Rand    : {ari_score:.4f}  (1.0=perfect cluster match)")
    print(f"{'='*45}\n")

    # Plots
    print("Generating plots...\n")
    plot_loss()
    plot_umap(embeddings, true_labels, pred_labels)
    plot_attention(attn_weights, true_labels)

    print("\n🎉 All done! Check outputs/ folder for:")
    print("   outputs/loss_curve.png")
    print("   outputs/umap_embeddings.png")
    print("   outputs/attention_weights.png")


if __name__ == '__main__':
    main()