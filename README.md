# HF Volume Prediction

DESCRIPTION
---

## 🔧 Project Structure

```text
hf-volume-prediction/
│
├── README.md   # Project overview, model description, instructions
│               
├── requirements/
│  ├── evn_export.ipynb 
│  └── requirements.txt # Python packages required
│        
├── LICENSE    
│                  # (e.g., MIT, Apache 2.0)
├── checkpoints/
│   ├── df_full_trx.parquet
│   ├── df_full_lob.parquet
│   └── best_model.pth         # trained model weights
│
├── data/                      # Input datasets (or scripts to download them)
│   └── preprocess.py          # Data preprocessing/feature extraction
│
├── models/                  # Model architecture
│   ├── garch.py             # Model architecture       
│   ├── gbm.py               # Model architecture
│   ├── tme.py               # Model architecture                  
│   ├── tme_ensamble.py      # Model architecture
│   └── attention_model.py
│
├── utils/   # Any function that is intended to be reused across multiple models or notebooks
│   └── pred_plot.py
│
├── notebooks/                 # Jupyter notebooks for visualization and EDA
│   ├── eda.ipynb
│   ├── garch_results.ipynb
│   ├── bgm_results.ipynb
│   ├── tme_results.ipynb
│   └── attention_model_results.ipynb
│
└── paper/                     # Optional NeurIPS paper source
    └── neurips_2025.tex
