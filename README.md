# Temporal Mixture Ensemble for Probabilistic Forecasting of Intraday Volume

This project reproduces and extends the *Temporal Mixture Ensemble (TME)* model for probabilistic forecasting of intraday trading volume in cryptocurrency markets, originally proposed by Antulov-Fantulin et al. (2020).

## Overview

We apply a probabilistic ensemble model that dynamically combines transaction and limit order book (LOB) data using a learned attention-like mechanism. The project includes:

- Re-implementation of ARMA-GARCH and XGBoost baselines
- Full reproduction of the TME model on Bitstamp BTC/USD 1-minute, 5-minute, and 10-minute data
- Extensions: 
  - SWAG (Stochastic Weight Averaging Gaussian)
  - Weighted ensemble averaging
  - Batch normalization
  - Alternative input preprocessing and training pipelines

## Features

- Ensemble-based probabilistic forecasting (mean + uncertainty)
- Multiple predictive horizons (1, 5, 10 min)
- Hyperparameter optimization with cross-validation
- Evaluation with RMSE, MAE, R², MAPE, NNLL, IW, and Coverage

## Results

We achieve improved predictive performance and calibration over baseline models. Our TME variant outperforms the original in some settings. See `Table 1` in the report for full results.

## Authors

- Benson Lee  
- Aliaksandr Samushchyk  
- Peter Furjesz  
ETH Zurich



        
