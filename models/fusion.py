import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.encoders import AcademicEncoder, BehavioralEncoder, ActivityEncoder


class CrossModalAttentionFusion(nn.Module):
    """
    Takes 3 embeddings (one per modality) and fuses them.
    
    Key idea: instead of treating all 3 equally, the model LEARNS
    how much to trust each modality per sample via attention weights.
    
    These attention weights are also your explainability output —
    they tell you which modality drove each prediction.
    """
    def __init__(self, embedding_dim=64, num_modalities=3, unified_dim=128):
        super().__init__()
        self.num_modalities = num_modalities

        # One attention scorer per modality
        # Input: all embeddings concatenated → Output: single score
        self.attention_heads = nn.ModuleList([
            nn.Linear(embedding_dim * num_modalities, 1)
            for _ in range(num_modalities)
        ])

        # Projects concatenated embeddings into final unified space
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim * num_modalities, unified_dim),
            nn.ReLU(),
            nn.LayerNorm(unified_dim)
        )

    def forward(self, embeddings):
        """
        Args:
            embeddings: list of 3 tensors, each [batch, embedding_dim]
        Returns:
            unified:      [batch, unified_dim]   ← the fused embedding
            attn_weights: [batch, 3]             ← explainability
        """
        # Concatenate all 3 embeddings → [B, 192]
        concat = torch.cat(embeddings, dim=-1)

        # Score each modality using the full concat context
        scores = torch.stack([
            head(concat) for head in self.attention_heads
        ], dim=1).squeeze(-1)                     # [B, 3]

        # Softmax so weights sum to 1 — comparable across samples
        attn_weights = F.softmax(scores, dim=-1)  # [B, 3]

        # Project to unified embedding space
        unified = self.projection(concat)         # [B, 128]

        return unified, attn_weights


class MultiModalFramework(nn.Module):
    """
    Full pipeline:
        raw features (3 modalities) → encoders → fusion → unified embedding
    """
    def __init__(self, embedding_dim=64, unified_dim=128):
        super().__init__()
        self.encoders = nn.ModuleDict({
            'academic':   AcademicEncoder(embedding_dim),
            'behavioral': BehavioralEncoder(embedding_dim),
            'activity':   ActivityEncoder(embedding_dim),
        })
        self.fusion = CrossModalAttentionFusion(
            embedding_dim=embedding_dim,
            num_modalities=3,
            unified_dim=unified_dim
        )

    def forward(self, academic, behavioral, activity):
        emb_a = self.encoders['academic'](academic)
        emb_b = self.encoders['behavioral'](behavioral)
        emb_c = self.encoders['activity'](activity)

        unified, attn = self.fusion([emb_a, emb_b, emb_c])
        return unified, attn


# ── Quick Test ─────────────────────────────────────────────────
if __name__ == '__main__':
    batch_size = 8
    model = MultiModalFramework(embedding_dim=64, unified_dim=128)

    # Dummy inputs — 8 people, 5 features each modality
    a = torch.randn(batch_size, 5)
    b = torch.randn(batch_size, 5)
    c = torch.randn(batch_size, 5)

    unified, attn = model(a, b, c)

    print("✅ Fusion Module works!\n")
    print(f"   Input per modality : {a.shape}")
    print(f"   Unified embedding  : {unified.shape}")    # [8, 128]
    print(f"   Attention weights  : {attn.shape}")       # [8, 3]

    print(f"\nAttention weights for each person (Academic | Behavioral | Activity):")
    print(f"(each row sums to 1.0)\n")
    for i in range(batch_size):
        w = attn[i].detach().numpy()
        print(f"   Person {i+1}: {w[0]:.3f}  |  {w[1]:.3f}  |  {w[2]:.3f}  → sum={w.sum():.3f}")

    # Count parameters
    total = sum(p.numel() for p in model.parameters())
    print(f"\nTotal model parameters: {total:,}")