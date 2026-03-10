import torch
import torch.nn as nn


class TabularEncoder(nn.Module):
    """
    Base encoder for any tabular/numerical modality.
    Takes raw features → compresses into a fixed-size embedding vector.
    """
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
            nn.LayerNorm(embedding_dim)         # keeps embedding values stable
        )

    def forward(self, x):
        return self.encoder(x)


class AcademicEncoder(TabularEncoder):
    """Encodes: gpa, attendance, assignment_completion, exam_avg, late_submissions"""
    def __init__(self, embedding_dim=64):
        super().__init__(input_dim=5, embedding_dim=embedding_dim)


class BehavioralEncoder(TabularEncoder):
    """Encodes: library_visits, session_duration, peer_interaction, forum_posts, login_variance"""
    def __init__(self, embedding_dim=64):
        super().__init__(input_dim=5, embedding_dim=embedding_dim)


class ActivityEncoder(TabularEncoder):
    """Encodes: steps, sleep_hours, active_minutes, sedentary_hours, resting_hr"""
    def __init__(self, embedding_dim=64):
        super().__init__(input_dim=5, embedding_dim=embedding_dim)


# ── Quick Test ─────────────────────────────────────────────────
if __name__ == '__main__':
    batch_size = 8

    print("Testing all 3 encoders...\n")

    for EncoderClass, name in [
        (AcademicEncoder,   "Academic"),
        (BehavioralEncoder, "Behavioral"),
        (ActivityEncoder,   "Activity"),
    ]:
        enc = EncoderClass(embedding_dim=64)
        dummy_input = torch.randn(batch_size, 5)      # 8 people, 5 features each
        output = enc(dummy_input)

        print(f"✅ {name} Encoder")
        print(f"   Input  shape : {dummy_input.shape}")   # [8, 5]
        print(f"   Output shape : {output.shape}")        # [8, 64]
        print(f"   Output range : [{output.min():.3f}, {output.max():.3f}]")
        print()

    print("All encoders working correctly!")