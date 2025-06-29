# HF Volume Prediction

DESCRIPTION
---




We propose a Temporal Mixture Ensemble (TME) model for probabilistic forecasting of intraday trading volume in cryptocurrency markets. Our model integrates temporal dynamics across multiple predictive components using a data-driven attention-like mixing mechanism. Evaluated on high-frequency Bitstamp exchange transaction and order book data. 


HF-Volume-Prediction/
├── README.md                   # Project overview, model description, setup instructions
├── main.pdf                   # Final report or paper
│
├── GBM_TME_SWAG/              # Experiments with GBM and TME (SWAG variant)
├── GARCH_TME_Improvements/    # Experiments and enhancements on GARCH and TME
│
├── TME_MAIN_ALL_FREQS/        # Core implementation
│   ├── requirements/
│   │   ├── env_export.ipynb   # Environment export script
│   │   └── requirements.txt   # Required Python packages
│   │
│   ├── checkpoints/
│   │   ├── df_full_trx.parquet   # Full transaction dataset
│   │   ├── df_full_lob.parquet   # Full order book dataset
│   │   └── best_model.pth        # Trained model weights
│   │
│   ├── data/
│   │   └── preprocess.py      # Data preprocessing and feature extraction
│   │
│   ├── models/
│   │   ├── tme_base.py              # Base TME model architecture
│   │   ├── tme_ensemble_cv.py       # Ensemble model with cross-validation
│   │   └── tme_hyper_param_tune.py  # Hyperparameter tuning logic
│   │
│   ├── utils/
│   │   └── pred_plot.py        # Reusable plotting functions
│   │
│   └── notebooks/
│       ├── ensemble_run.ipynb               # Main ensemble training
│       ├── ensemble_run_hyper_param.ipynb   # Hyperparameter tuning runs
│       ├── tme_run.ipynb                    # TME training script
│       └── distribution_test.ipynb          # Distributional comparison analysis

    

