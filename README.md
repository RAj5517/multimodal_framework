# Multi-Modal Representation Learning Framework

Unsupervised representation learning across heterogeneous data modalities using 
contrastive learning (SimCLR). Fuses academic performance, behavioral patterns, 
and real-world activity signals into unified embeddings for pattern discovery — 
without any labels.

**Key Result:** Adjusted Rand Index of 0.9989 — near-perfect unsupervised 
cluster recovery across 4 student profiles from 5000 samples.

---

## Problem Statement

Real-world data is heterogeneous. Academic records, app usage logs, and 
wearable sensor data all describe the same person but live in completely 
different feature spaces with different scales, dimensionalities, and 
noise profiles. Direct concatenation destroys this structure.

This framework solves the alignment problem: how do you take signals 
from fundamentally different sources and make them say something coherent 
about the same underlying entity?

This is identical to the sensor fusion problem in wearable health tech — 
fusing IMU, PPG, EMG, and EEG into a unified body state representation.

---

## Architecture
```
Academic [5]      Behavioral [5]      Activity [5]
     │                  │                  │
     ▼                  ▼                  ▼
AcademicEncoder   BehavioralEncoder  ActivityEncoder
  (5→128→64)        (5→128→64)         (5→128→64)
     │                  │                  │
     └──────────────────┴──────────────────┘
                        │
                        ▼
          CrossModalAttentionFusion
          - Concatenate: [64,64,64] → 192
          - Attention scores per modality (192→1 each)
          - Softmax → interpretable weights
          - Project: 192 → 128
                        │
                        ▼
              unified_embedding [128]
              attention_weights [3]   ← explainability
                        │
              (training only below)
                        ▼
              ProjectionHead (128→64)
                        │
                        ▼
                 NT-Xent Loss
```

### Why Each Component Exists

**Per-modality encoders** — Each modality needs its own encoder because 
raw features are incompatible. GPA (0-4) and heart rate (50-100) cannot 
share the same linear transformation meaningfully. Each encoder learns 
the internal structure of its own modality first.

**Cross-modal attention fusion** — Instead of treating all modalities 
equally, the model learns per-sample attention weights. If one modality 
is noisy or missing, the model down-weights it automatically. These 
weights also serve as your explainability output.

**Projection head** — A SimCLR technique. The encoder learns clean 
general representations. The projection head absorbs the contrastive 
task-specific distortion. The head is discarded after training; only 
the encoder is used at inference time.

---

## Training: SimCLR Contrastive Learning

Standard supervised learning requires labels. In most real-world sensor 
data — medical, industrial, behavioral — labels are expensive or 
unavailable. Contrastive learning solves this.

**Core idea:**
```
Same person + noise_v1  →  embedding z1  ┐
Same person + noise_v2  →  embedding z2  ┘ → push together
Different person        →  embedding z3    → push apart
```

**NT-Xent Loss** (Normalized Temperature-scaled Cross Entropy):
For each sample in a batch of N, identify its augmented pair 
from 2N-1 candidates. Temperature=0.07 makes the distribution 
sharp — the model must be confident, not just directionally correct.

**Key training decisions:**
- Batch size 128 — more negatives per step = stronger signal
- Warmup for 10 epochs — prevents bad early updates
- Cosine LR decay — smooth convergence
- Fixed noise augmentation (0.15) — curriculum noise caused training collapse
- Gradient clipping at 1.0 — prevents exploding gradients
- Early stopping with patience=30

---
## Model

Trained weights are available on HuggingFace:

**[raj5517/multimodal-representation-framework](https://huggingface.co/raj5517/multimodal-representation-framework)**
```python
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

weights = load_file(hf_hub_download(
    repo_id="raj5517/multimodal-representation-framework",
    filename="model.safetensors"
))
```
---

## Results

| Metric | Value | Meaning |
|--------|-------|---------|
| Final Loss | 0.5869 | NT-Xent contrastive loss |
| Silhouette Score | 0.3310 | Cluster separation quality |
| Adjusted Rand Index | **0.9989** | Cluster vs ground truth match |
| Training epochs | 184 (early stop) | Converged before max |
| Parameters | 66,243 (encoder) + 25,024 (projection) | Lightweight |

**ARI of 0.9989** means the model discovered all 4 student profiles 
from raw multimodal data with essentially zero errors — without ever 
seeing a single label during training.

---

## Explainability

The attention fusion module produces per-sample modality weights.

**Overall contribution across 5000 students:**
- Activity data: 49.1%
- Behavioral data: 29.1%  
- Academic data: 21.8%

**Per-profile insights:**
- Social Learner relies heavily on Activity (0.60) — physically active patterns
- High Achiever — balanced across all 3 modalities
- Quiet Worker relies on Behavioral (0.36) — library/forum patterns
- Struggling students show elevated Activity weight — health/lifestyle signal

---

## Project Structure
```
multimodal_framework/
├── data/
│   ├── generate_data.py      # Synthetic dataset with 4 hidden profiles
│   ├── academic.csv          # Modality 1 (not tracked in git)
│   ├── behavioral.csv        # Modality 2 (not tracked in git)
│   ├── activity.csv          # Modality 3 (not tracked in git)
│   └── labels.npy            # Ground truth for evaluation only
├── models/
│   ├── encoders.py           # Per-modality tabular encoders
│   ├── fusion.py             # Attention fusion + full framework
│   └── best_model.pt         # Saved weights (not tracked in git)
├── outputs/
│   ├── loss_curve.png        # Training loss over 184 epochs
│   ├── umap_embeddings.png   # 2D cluster visualization
│   └── attention_weights.png # Modality explainability plots
├── train.py                  # SimCLR training loop with GPU support
├── evaluate.py               # UMAP visualization + clustering metrics
└── requirements.txt
```

---

## Setup
```bash
git clone https://github.com/YOUR_USERNAME/multimodal_framework.git
cd multimodal_framework

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run
```bash
# Generate data
python data/generate_data.py

# Train (uses GPU automatically if available)
python train.py

# Visualize results
python evaluate.py
```

---

## Connection to Wearable Sensor Fusion

This framework directly addresses the multi-modal fusion problem in 
wearable health tech:

| This Project | Wearable Application |
|-------------|---------------------|
| Academic modality | EEG signals |
| Behavioral modality | EMG signals |
| Activity modality | IMU + PPG signals |
| Student profiles | Human activity/health states |
| Unsupervised clustering | Patient state discovery |
| Attention weights | Sensor reliability scoring |

The core challenge — heterogeneous signals with different sampling 
rates, units, and noise profiles — is identical. The architecture 
scales directly to raw time-series inputs by replacing the tabular 
encoders with 1D-CNN or LSTM encoders.

---

## Tech Stack

- PyTorch — model architecture and training
- SimCLR — contrastive self-supervised learning
- UMAP — non-linear dimensionality reduction for visualization
- KMeans — cluster assignment on learned embeddings
- SHAP — feature-level contribution analysis
- NVIDIA RTX 3050 — GPU accelerated training