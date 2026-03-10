---
license: mit
tags:
  - multimodal
  - representation-learning
  - contrastive-learning
  - simclr
  - unsupervised
  - pytorch
  - tabular
  - explainability
metrics:
  - adjusted_rand_score
  - silhouette_score
---

# Multi-Modal Representation Learning Framework

Unsupervised multi-modal representation learning framework that fuses
heterogeneous tabular signals into unified embeddings using cross-modal
attention and SimCLR contrastive training.

**Trained without any labels. Achieves ARI = 0.9989 on cluster recovery.**

---

## Model Architecture
```
Academic [5] + Behavioral [5] + Activity [5]
       ↓              ↓               ↓
  Encoder A      Encoder B       Encoder C
  (5→128→64)    (5→128→64)      (5→128→64)
       └──────────────┴───────────────┘
                       ↓
         CrossModalAttentionFusion
         - Concat [64,64,64] → 192
         - Per-modality attention scores
         - Softmax → weights sum to 1.0
         - Project 192 → 128
                       ↓
           unified_embedding [128]
           attention_weights [3]   ← explainability
```

- **Parameters:** 66,243 (encoder only)
- **Training:** SimCLR contrastive learning, 184 epochs, RTX 3050
- **Loss:** NT-Xent (temperature=0.07)
- **Batch size:** 128 with 256 negatives per step

---

## Results

| Metric | Score |
|--------|-------|
| NT-Xent Loss | 0.5869 |
| Silhouette Score | 0.3310 |
| **Adjusted Rand Index** | **0.9989** |

Near-perfect unsupervised cluster recovery across 4 student
profiles from 5000 samples — zero labels used during training.

---

## Quick Start
```python
import torch
from huggingface_hub import hf_hub_download
from modeling_multimodal import MultiModalFramework

# Load model
model = MultiModalFramework.from_pretrained("YOUR_HF_USERNAME/multimodal-representation-framework")
model.eval()

# Example: single student
academic   = torch.tensor([[3.7, 92.0, 90.0, 85.0, 1.0]])   # gpa, attendance%, assignment%, exam_avg, late
behavioral = torch.tensor([[5.0, 90.0, 6.0,  8.0,  2.0]])   # library, session_min, peer, forum, login_var
activity   = torch.tensor([[9000.0, 7.5, 60.0, 5.0, 62.0]]) # steps, sleep, active_min, sedentary, hr

with torch.no_grad():
    embedding, attn = model(academic, behavioral, activity)

print(f"Embedding shape : {embedding.shape}")        # [1, 128]
print(f"Attn weights    : {attn.numpy().round(3)}")  # [academic, behavioral, activity]
```

---

## Modality Attention Weights

The model produces per-sample attention weights explaining which
modality contributed most to the unified embedding.

**Overall contribution across 5000 students:**
- Activity: 49.1%
- Behavioral: 29.1%
- Academic: 21.8%

**Per-profile insights:**
- Social Learner relies heavily on Activity (0.60)
- Quiet Worker relies on Behavioral (0.36)
- High Achiever shows balanced attention across all modalities

---

## Application to Wearable Sensor Fusion

This framework directly addresses the multi-modal fusion problem in
wearable health tech. Replace tabular encoders with 1D-CNN/LSTM
encoders to handle:

| This Model | Wearable Application |
|-----------|---------------------|
| Academic modality | EEG signals |
| Behavioral modality | EMG signals |
| Activity modality | IMU + PPG |
| Student profiles | Human activity states |

---

## Training Details

- **Dataset:** Synthetic — 5000 samples, 4 hidden profiles
- **Augmentation:** Gaussian noise (σ=0.15) + 5% feature dropout
- **Optimizer:** Adam (lr=1e-3, weight_decay=1e-4)
- **LR Schedule:** 10-epoch warmup + cosine decay
- **Early stopping:** Patience=30
- **Hardware:** NVIDIA RTX 3050 4GB