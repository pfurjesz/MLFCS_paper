# HF Volume Prediction  
**Model**: Temporal Mixture Ensemble (TME) for cryptocurrency intraday volume forecasting.  
**Data**: High-frequency Bitstamp exchange transactions and order book data.  
**Key Features**:  
- Attention-like mixing mechanism.  
- Probabilistic forecasting.  

HF-Volume-Prediction/
├── README.md
├── main.pdf
├── GBM_TME_SWAG/
├── GARCH_TME_Improvements/
└── TME_MAIN_ALL_FREQS/
    ├── requirements/
    │   ├── env_export.ipynb
    │   └── requirements.txt
    ├── checkpoints/
    │   ├── df_full_trx.parquet
    │   ├── df_full_lob.parquet
    │   └── best_model.pth
    ├── data/
    │   └── preprocess.py
    ├── models/
    │   ├── tme_base.py
    │   ├── tme_ensemble_cv.py
    │   └── tme_hyper_param_tune.py
    ├── utils/
    │   └── pred_plot.py
    └── notebooks/
        ├── ensemble_run.ipynb
        ├── ensemble_run_hyper_param.ipynb
        ├── tme_run.ipynb
        └── distribution_test.ipynb
        
