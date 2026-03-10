"""
Multi-Modal Representation Learning Framework
Wraps the custom PyTorch architecture in a HuggingFace-compatible class.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin


# ── Encoder ────────────────────────────────────────────────────
class TabularEncoder(nn.Module):
    def __init__(self, input_dim, embedding_dim=64, hidden_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
    def forward(self, x):
        return self.encoder(x)

class AcademicEncoder(TabularEncoder):
    def __init__(self, embedding_dim=64):
        super().__init__(input_dim=5, embedding_dim=embedding_dim)

class BehavioralEncoder(TabularEncoder):
    def __init__(self, embedding_dim=64):
        super().__init__(input_dim=5, embedding_dim=embedding_dim)

class ActivityEncoder(TabularEncoder):
    def __init__(self, embedding_dim=64):
        super().__init__(input_dim=5, embedding_dim=embedding_dim)


# ── Fusion ─────────────────────────────────────────────────────
class CrossModalAttentionFusion(nn.Module):
    def __init__(self, embedding_dim=64, num_modalities=3, unified_dim=128):
        super().__init__()
        self.num_modalities = num_modalities
        self.attention_heads = nn.ModuleList([
            nn.Linear(embedding_dim * num_modalities, 1)
            for _ in range(num_modalities)
        ])
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim * num_modalities, unified_dim),
            nn.ReLU(),
            nn.LayerNorm(unified_dim)
        )

    def forward(self, embeddings):
        concat = torch.cat(embeddings, dim=-1)
        scores = torch.stack([
            head(concat) for head in self.attention_heads
        ], dim=1).squeeze(-1)
        attn_weights = F.softmax(scores, dim=-1)
        unified = self.projection(concat)
        return unified, attn_weights


# ── Full Model (HuggingFace compatible) ───────────────────────
class MultiModalFramework(nn.Module, PyTorchModelHubMixin):
    """
    Multi-Modal Representation Learning Framework.

    Fuses heterogeneous tabular signals into unified embeddings
    using cross-modal attention and SimCLR contrastive training.

    Inputs:
        academic   (B, 5): gpa, attendance_pct, assignment_completion,
                           exam_avg, late_submissions
        behavioral (B, 5): library_visits, session_duration,
                           peer_interaction, forum_posts, login_variance
        activity   (B, 5): steps_per_day, sleep_hours, active_minutes,
                           sedentary_hours, resting_hr

    Outputs:
        unified      (B, 128): fused embedding vector
        attn_weights (B, 3):   modality attention [academic, behavioral, activity]
    """

    def __init__(self, embedding_dim=64, unified_dim=128):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.unified_dim   = unified_dim

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

    def encode(self, academic, behavioral, activity):
        """Returns only the unified embedding. Use this for downstream tasks."""
        unified, _ = self.forward(academic, behavioral, activity)
        return unified

    def get_attention(self, academic, behavioral, activity):
        """Returns only attention weights. Use this for explainability."""
        _, attn = self.forward(academic, behavioral, activity)
        return attn