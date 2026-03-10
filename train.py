import sys, os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from models.fusion import MultiModalFramework

# ── Load & Normalize Data ──────────────────────────────────────
def load_data():
    academic   = pd.read_csv('data/academic.csv').values
    behavioral = pd.read_csv('data/behavioral.csv').values
    activity   = pd.read_csv('data/activity.csv').values

    # Normalize each modality independently (mean=0, std=1)
    normalized = []
    scalers = []
    for d in [academic, behavioral, activity]:
        sc = StandardScaler()
        normalized.append(sc.fit_transform(d))
        scalers.append(sc)

    tensors = [torch.FloatTensor(d) for d in normalized]
    return tensors, scalers


# ── Contrastive Loss (SimCLR style) ───────────────────────────
def nt_xent_loss(z1, z2, temperature=0.5):
    """
    Given two augmented views of the same data:
    - Pull same-person embeddings TOGETHER
    - Push different-person embeddings APART
    """
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    N  = z1.size(0)

    # Stack both views and compute all pairwise similarities
    representations = torch.cat([z1, z2], dim=0)       # [2N, dim]
    sim = F.cosine_similarity(
        representations.unsqueeze(1),
        representations.unsqueeze(0),
        dim=2
    ) / temperature                                     # [2N, 2N]

    # Labels: for each sample i in z1, its positive pair is i+N in z2
    labels = torch.arange(N, device=z1.device)
    labels = torch.cat([labels + N, labels])            # [2N]

    # Mask out self-similarity (diagonal)
    mask = torch.eye(2 * N, dtype=torch.bool, device=z1.device)
    sim.masked_fill_(mask, -1e9)

    loss = F.cross_entropy(sim, labels)
    return loss


# ── Data Augmentation ─────────────────────────────────────────
def add_noise(x, noise_level=0.1):
    """Add small random noise — creates two slightly different views of same data."""
    return x + noise_level * torch.randn_like(x)


# ── Training Loop ─────────────────────────────────────────────
def train():
    print("Loading data...")
    data, scalers = load_data()
    dataset = TensorDataset(*data)
    loader  = DataLoader(dataset, batch_size=64, shuffle=True)

    print("Building model...")
    model     = MultiModalFramework(embedding_dim=64, unified_dim=128)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}\n")
    print(f"{'Epoch':<8} {'Loss':<12} {'LR':<12}")
    print("-" * 35)

    EPOCHS = 50
    best_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0

        for batch in loader:
            a, b, c = batch

            # Create 2 augmented views of same batch
            unified1, _ = model(add_noise(a), add_noise(b), add_noise(c))
            unified2, _ = model(add_noise(a), add_noise(b), add_noise(c))

            loss = nt_xent_loss(unified1, unified2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(loader)
        lr       = optimizer.param_groups[0]['lr']

        if (epoch + 1) % 5 == 0:
            print(f"{epoch+1:<8} {avg_loss:<12.4f} {lr:<12.6f}")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'models/best_model.pt')

    print("\n✅ Training complete!")
    print(f"   Best loss      : {best_loss:.4f}")
    print(f"   Weights saved  : models/best_model.pt")
    return model, scalers


if __name__ == '__main__':
    model, scalers = train()