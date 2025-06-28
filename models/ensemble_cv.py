# =====================================================================
# TME Ensemble with Cross-Validation / I added cross val
# =====================================================================
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

warnings.filterwarnings('ignore')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================================
# Utility Functions Defined separately but I keep here as well
# =====================================================================
def deses(df: pd.DataFrame, time_col: str = 'datetime', 
          volume_col: str = 'total_volume', train_size: float = 0.8):
    """Deseasonalize volume data"""
    df = df.copy()
    df['time'] = df[time_col].dt.time
    train_idx = int(len(df)*train_size)
    train, rest = df.iloc[:train_idx].copy(), df.iloc[train_idx:].copy()
    
    mean_vol = train.groupby('time')[volume_col].mean().replace(0, 1e-8)
    
    for part in (train, rest):
        part['mean_volume'] = part['time'].map(mean_vol).fillna(part[volume_col])
        part['deseasoned_total_volume'] = part[volume_col] / part['mean_volume']
        part['log_deseasoned_total_volume'] = np.log(part['deseasoned_total_volume'].clip(lower=1e-8))
    
    return pd.concat([train, rest]), rest['mean_volume']

def rmse(y, y_hat): return np.sqrt(mean_squared_error(y, y_hat))
def mae(y, y_hat): return mean_absolute_error(y, y_hat)
def mape(y, y_hat): return np.mean(np.abs((y - y_hat) / np.clip(y, 1, None))) * 100
def iw(low, up): return np.mean(up - low)

def nnll(y_log, mu_log, sigma2_log):
    """Negative log likelihood normalized by number of points"""
    nll_each = 0.5 * (np.log(2*np.pi*sigma2_log) + (y_log - mu_log)**2 / sigma2_log)
    return nll_each.mean()

# Feature columns
TRX_COLS = ["buy_volume", "sell_volume", "buy_txn", "sell_txn",
            "volume_imbalance", "txn_imbalance"]
LOB_COLS = ["ask_volume", "bid_volume", "spread", "lob_volume_imbalance",
            "ask_slope_1", "ask_slope_5", "ask_slope_10",
            "bid_slope_1", "bid_slope_5", "bid_slope_10"]

# =====================================================================
# Dataset Class with Normalization
# =====================================================================
class WindowDS(Dataset):
    def __init__(self, df: pd.DataFrame, h: int, scaler_trx=None, scaler_lob=None, fit_scalers=False):
        self.time = df["datetime"].to_numpy()
        self.y = torch.tensor(df["log_deseasoned_total_volume"].to_numpy(np.float32))
        self.mv = torch.tensor(df["mean_volume"].to_numpy(np.float32))
        self.h = h
        
        if fit_scalers:
            self.scaler_trx = StandardScaler()
            self.scaler_lob = StandardScaler()
            self.scaler_trx.fit(df[TRX_COLS])
            self.scaler_lob.fit(df[LOB_COLS])
        else:
            self.scaler_trx = scaler_trx
            self.scaler_lob = scaler_lob
            
        self.trx = torch.tensor(self.scaler_trx.transform(df[TRX_COLS]), dtype=torch.float32)
        self.lob = torch.tensor(self.scaler_lob.transform(df[LOB_COLS]), dtype=torch.float32)
        
    def __len__(self): 
        return len(self.y) - self.h
        
    def __getitem__(self, i):
        sl = slice(i, i+self.h)
        return self.trx[sl], self.lob[sl], self.y[i+self.h], self.mv[i+self.h]

# =====================================================================
# Network Components (I fixed the Fixed Implementation)
# =====================================================================
class _Expert(nn.Module):
    """Expert network for one data source"""
    def __init__(self, d, h):
        super().__init__()
        # Mean parameters
        self.Lm = nn.Parameter(torch.randn(d*h, 1) * 0.01)
        self.Rm = nn.Parameter(torch.randn(1) * 0.01)
        self.bm = nn.Parameter(torch.zeros(1))
        
        # Variance parameters
        self.Lv = nn.Parameter(torch.randn(d*h, 1) * 0.01)
        self.Rv = nn.Parameter(torch.randn(1) * 0.01)
        self.bv = nn.Parameter(torch.zeros(1))
    

    # It can be used for all cases
    def _bilinear(self, x, L, R, b):
        """Bilinear transformation helper"""
        B = x.size(0)
        return (x.permute(0,2,1).reshape(B,-1) @ L).mul(R).sum(1, keepdim=True) + b

    #forward pass   
    def forward(self, x):
        mu = self._bilinear(x, self.Lm, self.Rm, self.bm)
        logvar = self._bilinear(x, self.Lv, self.Rv, self.bv)
        sigma2 = F.softplus(logvar) + 1e-5
        return mu, sigma2



# MAIN GATING - I fixed the Fixed Implementation
class _Gate(nn.Module):
    """Gating network"""
    def __init__(self, d_all, h, k=2):
        super().__init__()
        self.L = nn.Parameter(torch.randn(d_all*h, k) * 0.01)
        self.R = nn.Parameter(torch.randn(1, k) * 0.01)
        self.b = nn.Parameter(torch.zeros(k))
        
    def forward(self, *src):
        B = src[0].size(0)
        x = torch.cat([s.permute(0,2,1).reshape(B,-1) for s in src], 1)
        return torch.softmax((x @ self.L).mul(self.R) + self.b, 1)

# MAIN TME model, now its morte modular as discussed
class TME_Model(nn.Module):
    """Complete TME model"""
    def __init__(self, trx_dim, lob_dim, h):
        super().__init__()
        self.ex1 = _Expert(trx_dim, h)
        self.ex2 = _Expert(lob_dim, h)
        self.gate = _Gate(trx_dim + lob_dim, h)
        
    def forward(self, trx, lob):
        mu1, sigma2_1 = self.ex1(trx)
        mu2, sigma2_2 = self.ex2(lob)
        weights = self.gate(trx, lob)
        mu_combined = torch.cat([mu1, mu2], 1)
        sigma2_combined = torch.cat([sigma2_1, sigma2_2], 1)
        return mu_combined, sigma2_combined, weights

# =====================================================================
# TME Ensemble Class with Cross-Validation
# This is the new CV part, please read the paper to understand the general logic
# I am open to change it if needed. (uses the same config)
# =====================================================================
class TME_ensemble_CV:
    def __init__(self, df: pd.DataFrame, cfg: dict):
        self.cfg = cfg
        self.df = df.copy()
        self.h = cfg["model_params"].get("horizon", 100)
        self.n_splits = cfg["data_split"].get("n_splits", 5)
        self.test_size = cfg["data_split"].get("test_size", 0.2)
        
        # Initialize data structures
        self.models = []
        self.scalers = []
        self.fold_performance = []
        self.tr_curves = []
        self.va_curves = []
        
        # Split test set (holdout)
        test_idx = int(len(df) * (1 - self.test_size))
        self.df_train = df.iloc[:test_idx].copy()
        self.df_test = df.iloc[test_idx:].copy()
        
        # Initialize cross-validator
        self.tscv = TimeSeriesSplit(n_splits=self.n_splits)
    
    def train(self):
        """Train with cross-validation"""
        lr = self.cfg["model_params"]["learning_rate"]
        epochs = self.cfg["model_params"]["epochs"]
        bs = self.cfg["model_params"]["batch_size"]
        n_models = self.cfg["model_params"].get("n_models", 10)
        l2_lambda = self.cfg["model_params"].get("l2_lambda", 0.01)
        
        # Outer CV loop
        for fold, (train_idx, val_idx) in enumerate(self.tscv.split(self.df_train)):
            print(f"\n=== Training Fold {fold+1}/{self.n_splits} ===")
            
            # Split data
            df_fold_train = self.df_train.iloc[train_idx].copy()
            df_fold_val = self.df_train.iloc[val_idx].copy()
            
            # Fit scalers for this fold
            scaler_trx = StandardScaler().fit(df_fold_train[TRX_COLS])
            scaler_lob = StandardScaler().fit(df_fold_train[LOB_COLS])
            self.scalers.append((scaler_trx, scaler_lob))
            
            # Create datasets
            train_ds = WindowDS(df_fold_train, self.h, scaler_trx, scaler_lob)
            val_ds = WindowDS(df_fold_val, self.h, scaler_trx, scaler_lob)
            
            train_dl = DataLoader(train_ds, bs, shuffle=True)
            val_dl = DataLoader(val_ds, bs, shuffle=False)
            
            fold_models = []
            fold_performance = []
            
            # Train ensemble for this fold
            for i in range(n_models):
                print(f"\nTraining model {i+1}/{n_models}")
                model = TME_Model(len(TRX_COLS), len(LOB_COLS), self.h).double().to(DEVICE)
                opt = torch.optim.Adam(model.parameters(), lr=lr)
                
                best_val, best_state = np.inf, None
                tr_curve, va_curve = [], []
                
                for ep in range(1, epochs+1):
                    # Training
                    model.train()
                    train_loss = 0.
                    for trx, lob, y, _ in train_dl:
                        trx, lob, y = [t.double().to(DEVICE) for t in (trx, lob, y)]
                        opt.zero_grad()
                        mu, sigma2, w = model(trx, lob)
                        loss = self._nll(y, mu, sigma2, w)
                        
                        if l2_lambda > 0:
                            l2_reg = sum(p.pow(2).sum() for p in model.parameters())
                            loss = loss + l2_lambda * l2_reg
                        
                        loss.backward()
                        opt.step()
                        train_loss += loss.item()
                    
                    tr_curve.append(train_loss/len(train_dl))
                    
                    # Validation
                    model.eval()
                    val_loss = 0.
                    with torch.no_grad():
                        for trx, lob, y, _ in val_dl:
                            trx, lob, y = [t.double().to(DEVICE) for t in (trx, lob, y)]
                            mu, sigma2, w = model(trx, lob)
                            val_loss += self._nll(y, mu, sigma2, w).item()
                    
                    val_loss /= len(val_dl)
                    va_curve.append(val_loss)
                    
                    if val_loss < best_val:
                        best_val = val_loss
                        best_state = model.state_dict()
                    
                    print(f"ep{ep:02d}  train {tr_curve[-1]:.4f}  val {val_loss:.4f}")
                
                # Save best model for this fold
                model.load_state_dict(best_state)
                fold_models.append(model)
                fold_performance.append(best_val)
                self.tr_curves.append(tr_curve)
                self.va_curves.append(va_curve)
            
            self.models.append(fold_models)
            self.fold_performance.append(fold_performance)
        
        print("\nTraining completed across all folds!")
    

    # I dont include the reg here added during the training (or included in the optimizer)
    def _nll(self, y, mu, sigma2, w):
        """Negative log likelihood for a single model"""
        log_prob = -0.5 * (torch.log(2 * np.pi * sigma2) + (y.unsqueeze(1) - mu)**2 / sigma2)
        return -torch.logsumexp(torch.log(w.clamp(1e-4, 1-1e-4)) + log_prob, 1).mean()
    
    def evaluate(self):
        """Evaluate on test set using all models from all folds"""
        # Create test dataset using last fold's scalers
        scaler_trx, scaler_lob = self.scalers[-1]
        test_ds = WindowDS(self.df_test, self.h, scaler_trx, scaler_lob)
        test_dl = DataLoader(test_ds, self.cfg["model_params"]["batch_size"], shuffle=False)
        
        y_log_all, mu_log_all, sigma2_log_all, mv_all = [], [], [], []
        
        # Get predictions from all models across all folds (THIS IS THE KEY PART, we predict
        # with all the models and then average them)
        for fold_models in self.models:
            for model in fold_models:
                model.eval()
                y_log, mu_log, sigma2_log, mv = [], [], [], []
                
                with torch.no_grad():
                    for trx, lob, y, m in test_dl:
                        trx, lob, y = [t.double().to(DEVICE) for t in (trx, lob, y)]
                        mu_k, sigma2_k, w = model(trx, lob)
                        
                        # Combine predictions
                        mu = (w * mu_k).sum(1)
                        sigma2 = (w * sigma2_k).sum(1) + (w * ((mu_k - mu[:,None])**2)).sum(1)
                        
                        y_log.append(y.cpu().numpy())
                        mu_log.append(mu.cpu().numpy())
                        sigma2_log.append(sigma2.cpu().numpy())
                        mv.append(m.numpy())
                
                y_log_all.append(np.concatenate(y_log))
                mu_log_all.append(np.concatenate(mu_log))
                sigma2_log_all.append(np.concatenate(sigma2_log))
                mv_all.append(np.concatenate(mv))
        
        # Stack predictions
        y_log = np.stack(y_log_all)
        mu_log = np.stack(mu_log_all)
        sigma2_log = np.stack(sigma2_log_all)
        mv = np.stack(mv_all)
        
        # Compute ensemble statistics
        mu_log_ensemble = mu_log.mean(0)
        sigma2_log_ensemble = (sigma2_log + mu_log**2).mean(0) - mu_log_ensemble**2
        
        # Convert to original scale (added clip to avoid overflow)
        y_true = np.exp(np.clip(y_log[0], -700, 700)) * mv[0]
        y_pred = np.exp(np.clip(mu_log_ensemble, -700, 700)) * mv[0]
        
        # Compute prediction intervals
        sigma2_log_clip = np.clip(sigma2_log_ensemble, -700, 700)
        mu_log_clip = np.clip(mu_log_ensemble, -700, 700)
        
        exp_sigma2 = np.exp(sigma2_log_clip)
        exp_2mu_plus_sigma2 = np.exp(2 * mu_log_clip + sigma2_log_clip)
        
        safe_exp = np.where(exp_sigma2 > 1e300, 1e300, exp_sigma2)
        safe_term = (safe_exp - 1) * np.where(exp_2mu_plus_sigma2 > 1e300, 1e300, exp_2mu_plus_sigma2)
        
        sigma = np.sqrt(np.clip(safe_term, 0, 1e300)) * mv[0]
        low, up = y_pred - 0.5*sigma, y_pred + 0.5*sigma

        # Clip values for metrics
        y_true_clip = np.clip(y_true, 0, 1e10)
        y_pred_clip = np.clip(y_pred, 0, 1e10)
        low = np.clip(low, 0, 1e10)
        up = np.clip(up, 0, 1e10)

        # Plot results
        test_datetimes = self.df_test["datetime"].iloc[self.h:].values
        
        plt.figure(figsize=(12, 4))
        plt.fill_between(
            test_datetimes,
            np.clip(low, 0, 5000), 
            np.clip(up, 0, 5000), 
            color="#9ecae1", 
            alpha=0.3, 
            label="95% band"
        )
        plt.plot(test_datetimes, y_true_clip, label="true", lw=0.7, color="#4daf4a")
        plt.plot(test_datetimes, y_pred_clip, label="pred", lw=0.7, color="#e41a1c")
        
        plt.gcf().autofmt_xdate()
        plt.xlabel("Datetime")
        plt.ylabel("Volume")
        plt.title(f"Test Set Predictions (Avg of {len(self.models)*len(self.models[0])} models)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        # Print metrics
        print(f"{'Metric':<10} | {'Value':>10}")
        print("-" * 23)
        print(f"{'RMSE':<10}   | {rmse(y_true_clip, y_pred_clip):10.2f}")
        print(f"{'MAE':<10}    | {mae(y_true_clip, y_pred_clip):10.2f}")
        print(f"{'R²':<10}     | {r2_score(y_true_clip, y_pred_clip):10.4f}")
        print(f"{'MAPE (%)':<10}| {mape(y_true_clip, y_pred_clip):10.2f}")
        print(f"{'Coverage':<10}| {np.mean((y_true_clip >= low) & (y_true_clip <= up)) * 100:10.1f} %")
        print(f"{'IW':<10}     | {iw(low, up):10.2f}")
        print(f"{'NNLL':<10}   | {nnll(y_log[0], mu_log_ensemble, sigma2_log_ensemble):10.4f}")
        
        return y_true_clip, y_pred_clip
    
    def plot_learning_curves(self):
        """Plot learning curves across all folds"""
        plt.figure(figsize=(12, 6))
        
        # Convert to numpy arrays
        tr_curves = np.array(self.tr_curves)
        va_curves = np.array(self.va_curves)
        
        # Plot individual curves
        for i in range(len(tr_curves)):
            plt.plot(tr_curves[i], color='blue', alpha=0.1, linewidth=1)
            plt.plot(va_curves[i], color='orange', alpha=0.1, linewidth=1)
        
        # Plot average curves
        plt.plot(tr_curves.mean(0), color='blue', linewidth=3, label='Avg Training')
        plt.plot(va_curves.mean(0), color='orange', linewidth=3, label='Avg Validation')
        
        plt.title('Learning Curves Across All Folds and Models')
        plt.xlabel('Epoch')
        plt.ylabel('Negative Log Likelihood')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_variable_importance(self):
        """Plot variable importance across all models"""
        trx_importance = {col: 0 for col in TRX_COLS}
        lob_importance = {col: 0 for col in LOB_COLS}
        
        for fold_models in self.models:
            for model in fold_models:
                with torch.no_grad():
                    # Transaction features
                    for i, col in enumerate(TRX_COLS):
                        for t in range(self.h):
                            trx_importance[col] += torch.abs(model.ex1.Lm[i*self.h + t]).item()
                    
                    # Order book features
                    for i, col in enumerate(LOB_COLS):
                        for t in range(self.h):
                            lob_importance[col] += torch.abs(model.ex2.Lm[i*self.h + t]).item()
        
        # Normalize
        total_trx = sum(trx_importance.values()) or 1e-8
        total_lob = sum(lob_importance.values()) or 1e-8
        
        trx_importance = {k: v/total_trx for k, v in trx_importance.items()}
        lob_importance = {k: v/total_lob for k, v in lob_importance.items()}
        
        # Create plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        trx_df = pd.DataFrame.from_dict(trx_importance, orient='index', columns=['Importance'])
        trx_df.sort_values('Importance', ascending=True).plot(kind='barh', ax=ax1)
        ax1.set_title('Transaction Features Importance')
        ax1.set_xlabel('Normalized Importance Score')
        
        lob_df = pd.DataFrame.from_dict(lob_importance, orient='index', columns=['Importance'])
        lob_df.sort_values('Importance', ascending=True).plot(kind='barh', ax=ax2)
        ax2.set_title('Order Book Features Importance')
        ax2.set_xlabel('Normalized Importance Score')
        
        plt.tight_layout()
        plt.show()
    
    def save(self, path):
        """Save the entire ensemble"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state_dicts = [[m.state_dict() for m in fold] for fold in self.models]
        torch.save({
            'state_dicts': state_dicts,
            'scalers': self.scalers,
            'cfg': self.cfg,
            'h': self.h,
            'tr_curves': self.tr_curves,
            'va_curves': self.va_curves,
            'fold_performance': self.fold_performance
        }, path)
        print(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path, df):
        """Load a saved ensemble"""
        data = torch.load(path)
        ensemble = cls(df, data['cfg'])
        ensemble.h = data['h']
        
        # Reconstruct models
        ensemble.models = []
        for fold_state_dicts in data['state_dicts']:
            fold_models = []
            for state_dict in fold_state_dicts:
                model = TME_Model(len(TRX_COLS), len(LOB_COLS), ensemble.h).double().to(DEVICE)
                model.load_state_dict(state_dict)
                fold_models.append(model)
            ensemble.models.append(fold_models)
        
        ensemble.scalers = data['scalers']
        ensemble.tr_curves = data['tr_curves']
        ensemble.va_curves = data['va_curves']
        ensemble.fold_performance = data['fold_performance']
        print(f"Model loaded from {path}")
        return ensemble


# DONE:)

