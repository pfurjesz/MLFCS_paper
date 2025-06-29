# HF Volume Prediction

DESCRIPTION
---




We propose a Temporal Mixture Ensemble (TME) model for probabilistic forecasting of intraday trading volume in cryptocurrency markets. Our model integrates temporal dynamics across multiple predictive components using a data-driven attention-like mixing mechanism. Evaluated on high-frequency Bitstamp exchange transaction and order book data. 


## 🔧 Project Structure
├── README.md   # Project overview, model description, instructions
├── main.pdf     
├── GBM, TME (SWAG)
├── GARCH, TME Improvments
├── main.pdf 
└── TME_MIAN_ALL_FREQS/          
    ├── requirements/
    │  ├── evn_export.ipynb 
    │  └── requirements.txt # Python packages required
    │        
    ├── checkpoints/
    │   ├── df_full_trx.parquet
    │   ├── df_full_lob.parquet
    │   └── best_model.pth         # trained model weights
    │
    ├── data/                      # Input datasets (or scripts to download them)
    │   └── preprocess.py          # Data preprocessing/feature extraction
    │
    ├── models/                  # Model architecture
    │   ├── tme_base.py               # Model architecture                  
    │   ├── tme_ensamble_cv.py      # Model architecture
    │   └── tme_hyper_param_tune.py
    │
    ├── utils/   # Any function that is intended to be reused across multiple models or notebooks
    │   └── pred_plot.py
    │
    ├── notebooks/                 # Jupyter notebooks for visualization and EDA
        ├── ensemble_run_hyper_param.ipynb
        ├── ensemble_run.ipynb
        ├── tme_run.ipynb
        └── distribution_test.ipynb
        
    
    

