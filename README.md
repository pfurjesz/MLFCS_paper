# HF Volume Prediction

This repository contains code and data pipelines for predicting high-frequency (HF) trading volume using deep learning models such as LSTMs. The project is structured for clarity, modularity, and reproducibility, and is designed with NeurIPS-style standards in mind.

---

## 🔧 Project Structure

```text
hf-volume-prediction/
│
├── README.md                  # Project overview, model description, instructions
├── requirements.txt           # Python packages required
├── LICENSE                    # (e.g., MIT, Apache 2.0)
├── checkpoints/
│   └── best_model.pth         # trained model weights
│
├── config.yaml                # Model and training configuration
├── data/                      # Input datasets (or scripts to download them)
│   └── preprocess.py          # Data preprocessing/feature extraction
│
├── src/                       # Core source code
│   ├── model.py               # Model architecture
│   ├── train.py               # Training loop
│   ├── evaluate.py            # Evaluation logic
│   ├── utils.py               # Utility functions (e.g., metrics, logging)
│   └── config.py              # Central config loader (optional)
│
├── scripts/                   # Scripts to run training/evaluation
│   ├── run_train.py
│   └── run_eval.py
│
├── notebooks/                 # Jupyter notebooks for visualization and EDA
│   └── eda.ipynb
│
├── results/                   # Prediction outputs and plots
│   └── test_preds.csv
│
└── paper/                     # Optional NeurIPS paper source
    └── neurips_2025.tex
