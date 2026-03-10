# Multi-Modal Representation Learning Framework

Fuses heterogeneous signals (academic, behavioral, activity) into unified embeddings for unsupervised pattern discovery.

## Project Structure
```
multimodal_framework/
├── data/               # Raw + generated datasets (not tracked in git)
├── models/             # Model architecture files
├── outputs/            # Plots and visualizations
├── notebooks/          # Jupyter demos
├── train.py            # Training script
├── evaluate.py         # Evaluation + clustering
└── requirements.txt    # Dependencies
```

## Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
python data/generate_data.py   # Step 1: Generate data
python models/encoders.py      # Step 2: Test encoders
python models/fusion.py        # Step 3: Test fusion
python train.py                # Step 4: Train
python evaluate.py             # Step 5: Visualize
```
