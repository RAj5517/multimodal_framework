import torch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from huggingface_hub import HfApi, create_repo
from safetensors.torch import save_file
from hf_upload.modeling_multimodal import MultiModalFramework

HF_USERNAME = "raj5517"
REPO_NAME   = "multimodal-representation-framework"
REPO_ID     = f"{HF_USERNAME}/{REPO_NAME}"
SAVE_DIR    = "hf_upload/model_files"

os.makedirs(SAVE_DIR, exist_ok=True)

# ── Create repo ────────────────────────────────────────────────
print(f"Creating repo: {REPO_ID}")
create_repo(REPO_ID, exist_ok=True, repo_type="model")

# ── Load weights ───────────────────────────────────────────────
print("Loading weights...")
model      = MultiModalFramework(embedding_dim=64, unified_dim=128)
checkpoint = torch.load(
    'models/best_model.pt',
    map_location='cpu',
    weights_only=True       # fixes FutureWarning
)
model.load_state_dict(checkpoint['model'])
model.eval()

# ── Save as safetensors ────────────────────────────────────────
print("Saving as safetensors...")
tensors = {k: v.contiguous() for k, v in model.state_dict().items()}
save_file(tensors, f"{SAVE_DIR}/model.safetensors")
print(f"✅ Saved: {SAVE_DIR}/model.safetensors")

# ── Save config ────────────────────────────────────────────────
import json
config = {
    "model_type": "multimodal_framework",
    "embedding_dim": 64,
    "unified_dim": 128,
    "num_modalities": 3,
    "modalities": ["academic", "behavioral", "activity"],
    "input_dims": {"academic": 5, "behavioral": 5, "activity": 5},
    "training": {
        "method": "SimCLR contrastive learning",
        "epochs": 184,
        "best_loss": 0.5869,
        "temperature": 0.07,
        "batch_size": 128,
        "hardware": "NVIDIA RTX 3050 4GB"
    },
    "metrics": {
        "silhouette_score": 0.3310,
        "adjusted_rand_index": 0.9989
    }
}
with open(f"{SAVE_DIR}/config.json", "w") as f:
    json.dump(config, f, indent=2)
print("✅ Saved: config.json")

# ── Upload everything ──────────────────────────────────────────
print("\nUploading to HuggingFace...")
api = HfApi()

# Model weights
api.upload_file(
    path_or_fileobj=f"{SAVE_DIR}/model.safetensors",
    path_in_repo="model.safetensors",
    repo_id=REPO_ID,
)
print("✅ Uploaded: model.safetensors")

# Config
api.upload_file(
    path_or_fileobj=f"{SAVE_DIR}/config.json",
    path_in_repo="config.json",
    repo_id=REPO_ID,
)
print("✅ Uploaded: config.json")

# Model architecture code
api.upload_file(
    path_or_fileobj="hf_upload/modeling_multimodal.py",
    path_in_repo="modeling_multimodal.py",
    repo_id=REPO_ID,
)
print("✅ Uploaded: modeling_multimodal.py")

# Model card
api.upload_file(
    path_or_fileobj="hf_upload/README.md",
    path_in_repo="README.md",
    repo_id=REPO_ID,
)
print("✅ Uploaded: README.md")

print(f"\n🎉 Done!")
print(f"   View at: https://huggingface.co/{REPO_ID}")