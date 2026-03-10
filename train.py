import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from models.fusion import MultiModalFramework

# ── Device ────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")


# ── Projection Head ───────────────────────────────────────────
class ProjectionHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128, output_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
    def forward(self, x):
        return self.net(x)


# ── NT-Xent Loss ───────────────────────────────────────────────
def nt_xent_loss(z1, z2, temperature=0.07):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    N  = z1.size(0)

    representations = torch.cat([z1, z2], dim=0)
    sim = F.cosine_similarity(
        representations.unsqueeze(1),
        representations.unsqueeze(0),
        dim=2
    ) / temperature

    labels = torch.arange(N, device=z1.device)
    labels = torch.cat([labels + N, labels])

    mask = torch.eye(2 * N, dtype=torch.bool, device=z1.device)
    sim.masked_fill_(mask, -1e9)

    return F.cross_entropy(sim, labels)


# ── Fixed Augmentation (no curriculum — was causing explosion) ─
def augment(x, noise_level=0.15):
    """
    Fixed noise level — stable throughout training.
    Curriculum noise was destroying signal after epoch 60.
    """
    noisy = x + noise_level * torch.randn_like(x)
    # Light feature dropout only
    mask  = (torch.rand_like(x) > 0.05).float()   # only 5% dropout
    return noisy * mask


# ── LR Warmup + Cosine Decay ───────────────────────────────────
def get_lr(epoch, warmup=10, max_lr=1e-3, min_lr=1e-5, total=200):
    if epoch < warmup:
        return max_lr * (epoch + 1) / warmup
    p = (epoch - warmup) / (total - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos(np.pi * p))


# ── Load Data ─────────────────────────────────────────────────
def load_data():
    academic   = pd.read_csv('data/academic.csv').values
    behavioral = pd.read_csv('data/behavioral.csv').values
    activity   = pd.read_csv('data/activity.csv').values

    normalized, scalers = [], []
    for d in [academic, behavioral, activity]:
        sc = StandardScaler()
        normalized.append(sc.fit_transform(d))
        scalers.append(sc)

    return [torch.FloatTensor(d) for d in normalized], scalers


# ── Train ──────────────────────────────────────────────────────
def train():
    EPOCHS     = 200
    BATCH_SIZE = 128       # GPU can handle bigger batch = more negatives
    TEMP       = 0.07
    PATIENCE   = 30        # early stopping

    print("\nLoading data...")
    data, scalers = load_data()
    dataset = TensorDataset(*data)
    loader  = DataLoader(
        dataset, batch_size=BATCH_SIZE,
        shuffle=True, drop_last=True,
        pin_memory=(device.type == 'cuda')   # faster GPU transfer
    )

    print("Building model...")
    model     = MultiModalFramework(embedding_dim=64, unified_dim=128).to(device)
    proj_head = ProjectionHead(input_dim=128, hidden_dim=128, output_dim=64).to(device)

    all_params = list(model.parameters()) + list(proj_head.parameters())
    optimizer  = torch.optim.Adam(all_params, lr=1e-3, weight_decay=1e-4)

    print(f"Encoder params     : {sum(p.numel() for p in model.parameters()):,}")
    print(f"Projection params  : {sum(p.numel() for p in proj_head.parameters()):,}")
    print(f"Batch size         : {BATCH_SIZE} ({BATCH_SIZE*2} negatives per step)")
    print(f"Temperature        : {TEMP}")
    print(f"Epochs             : {EPOCHS}")
    print(f"Early stop patience: {PATIENCE}\n")

    print(f"{'Epoch':<8} {'Loss':<12} {'LR':<12} {'Best':<10} {'Status'}")
    print("-" * 55)

    best_loss     = float('inf')
    patience_count = 0
    loss_history  = []

    for epoch in range(EPOCHS):
        model.train()
        proj_head.train()

        lr = get_lr(epoch, warmup=10, max_lr=1e-3, total=EPOCHS)
        for g in optimizer.param_groups:
            g['lr'] = lr

        epoch_loss = 0
        for batch in loader:
            a, b, c = [x.to(device) for x in batch]

            unified1, _ = model(augment(a), augment(b), augment(c))
            unified2, _ = model(augment(a), augment(b), augment(c))

            z1 = proj_head(unified1)
            z2 = proj_head(unified2)

            loss = nt_xent_loss(z1, z2, temperature=TEMP)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        loss_history.append(avg_loss)

        # Early stopping
        status = ""
        if avg_loss < best_loss:
            best_loss      = avg_loss
            patience_count = 0
            status         = "✅ saved"
            torch.save({
                'model':     model.state_dict(),
                'proj_head': proj_head.state_dict(),
                'epoch':     epoch + 1,
                'loss':      best_loss,
            }, 'models/best_model.pt')
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"\n⏹ Early stopping at epoch {epoch+1}")
                break

        if (epoch + 1) % 10 == 0 or status:
            print(f"{epoch+1:<8} {avg_loss:<12.4f} {lr:<12.6f} {best_loss:<10.4f} {status}")

    print(f"\n✅ Training complete!")
    print(f"   Best loss : {best_loss:.4f}")
    print(f"   Saved     : models/best_model.pt")

    np.save('outputs/loss_history.npy', np.array(loss_history))
    return model, proj_head, scalers


if __name__ == '__main__':
    model, proj_head, scalers = train()