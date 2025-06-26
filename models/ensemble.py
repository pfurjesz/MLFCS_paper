# =====================================================================
# Final Corrected TME Implementation
# =====================================================================
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================================
# Utility Functions
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
        
        # Initialize or use existing scalers
        if fit_scalers:
            self.scaler_trx = StandardScaler()
            self.scaler_lob = StandardScaler()
            self.scaler_trx.fit(df[TRX_COLS])
            self.scaler_lob.fit(df[LOB_COLS])
        else:
            self.scaler_trx = scaler_trx
            self.scaler_lob = scaler_lob
            
        # Scale features
        self.trx = torch.tensor(self.scaler_trx.transform(df[TRX_COLS]), dtype=torch.float32)
        self.lob = torch.tensor(self.scaler_lob.transform(df[LOB_COLS]), dtype=torch.float32)
        
    def __len__(self): 
        return len(self.y) - self.h
        
    def __getitem__(self, i):
        sl = slice(i, i+self.h)
        return self.trx[sl], self.lob[sl], self.y[i+self.h], self.mv[i+self.h]

# =====================================================================
# Network Components with L2 Regularization
# =====================================================================
def _bilinear(x, L, R, b):
    """Bilinear transformation"""
    B = x.size(0)
    return (x.permute(0,2,1).reshape(B,-1) @ L).mul(R).sum(1, keepdim=True) + b

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
        
    def forward(self, x):
        mu = _bilinear(x, self.Lm, self.Rm, self.bm)
        logvar = _bilinear(x, self.Lv, self.Rv, self.bv)
        sigma2 = F.softplus(logvar) + 1e-5
        return mu, sigma2

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

# =====================================================================
# TME Ensemble Class with All Improvements
# =====================================================================
class TME_ensemble:
    def __init__(self, df: pd.DataFrame, cfg: dict):
        self.cfg = cfg
        train_q, val_q = cfg["data_split"]["train_size"], cfg["data_split"]["validation_size"]
        n = len(df)
        i1 = int(train_q * n)
        i2 = int((train_q + val_q) * n)
        
        # Initialize scalers
        self.scaler_trx = StandardScaler()
        self.scaler_lob = StandardScaler()
        
        # Prepare data with normalization
        self.df_train = df.iloc[:i1].copy()
        self.df_val = df.iloc[i1:i2].copy()
        self.df_test = df.iloc[i2:].copy()
        
        # Fit scalers on training data
        self.scaler_trx.fit(self.df_train[TRX_COLS])
        self.scaler_lob.fit(self.df_train[LOB_COLS])
        
        h = cfg["model_params"].get("horizon", 100)
        bs = cfg["model_params"]["batch_size"]
        
        # Create datasets with normalization
        self.train_ds = WindowDS(self.df_train, h, self.scaler_trx, self.scaler_lob)
        self.val_ds = WindowDS(self.df_val, h, self.scaler_trx, self.scaler_lob)
        self.test_ds = WindowDS(self.df_test, h, self.scaler_trx, self.scaler_lob)
        
        self.train_dl = DataLoader(self.train_ds, bs, shuffle=True)
        self.val_dl = DataLoader(self.val_ds, bs, shuffle=False)
        self.test_dl = DataLoader(self.test_ds, bs, shuffle=False)
        
        self.n_models = cfg["model_params"].get("n_models", 20)
        self.l2_lambda = cfg["model_params"].get("l2_lambda", 0.01)
        self.models = []
        self.model_performance = []
        self.h = h
        self.tr_curves = []
        self.va_curves = []
        self.selected_models = []
    
    def _create_model(self):
        """Create a single TME model instance"""
        model = nn.Module()
        model.ex1 = _Expert(len(TRX_COLS), self.h)
        model.ex2 = _Expert(len(LOB_COLS), self.h)
        model.gate = _Gate(len(TRX_COLS)+len(LOB_COLS), self.h)
        return model.to(DEVICE).double()
    
    def train(self):
        """Train ensemble of models and keep only top 50%"""
        lr = self.cfg["model_params"]["learning_rate"]
        epochs = self.cfg["model_params"]["epochs"]
        
        for i in range(self.n_models):
            print(f"\nTraining model {i+1}/{self.n_models}")
            model = self._create_model()
            opt = torch.optim.Adam(model.parameters(), lr=lr)
            
            best_val, best_state = np.inf, None
            tr_curve, va_curve = [], []
            
            for ep in range(1, epochs+1):
                # Training
                model.train()
                train_loss = 0.
                for trx, lob, y, _ in self.train_dl:
                    trx, lob, y = [t.double().to(DEVICE) for t in (trx, lob, y)]
                    opt.zero_grad()
                    mu, sigma2, w = self._fwd(model, trx, lob)
                    loss = self._nll(y, mu, sigma2, w)
                    
                    # Add L2 regularization
                    if self.l2_lambda > 0:
                        l2_reg = sum(p.pow(2).sum() for p in model.parameters())
                        loss = loss + self.l2_lambda * l2_reg
                    
                    loss.backward()
                    opt.step()
                    train_loss += loss.item()
                tr_curve.append(train_loss/len(self.train_dl))
                
                # Validation
                model.eval()
                val_loss = 0.
                with torch.no_grad():
                    for trx, lob, y, _ in self.val_dl:
                        trx, lob, y = [t.double().to(DEVICE) for t in (trx, lob, y)]
                        mu, sigma2, w = self._fwd(model, trx, lob)
                        val_loss += self._nll(y, mu, sigma2, w).item()
                val_loss /= len(self.val_dl)
                va_curve.append(val_loss)
                
                if val_loss < best_val:
                    best_val = val_loss
                    best_state = model.state_dict()
                
                print(f"ep{ep:02d}  train {tr_curve[-1]:.4f}  val {val_loss:.4f}")
            
            # Save curves and best model
            self.tr_curves.append(tr_curve)
            self.va_curves.append(va_curve)
            model.load_state_dict(best_state)
            self.models.append(model)
            self.model_performance.append(best_val)
        
        # After all models trained, select top 50%
        self._select_best_models()
    
    def _select_best_models(self):
        """Select top 50% of models based on validation performance"""
        n_keep = max(1, int(self.n_models * 0.5))
        self.model_performance = np.array(self.model_performance)
        sorted_indices = np.argsort(self.model_performance)
        self.selected_models = sorted_indices[:n_keep]
        
        print(f"\nSelected {n_keep} best models out of {self.n_models}")
        print("Validation performance of selected models:")
        for idx in self.selected_models:
            print(f"Model {idx+1}: {self.model_performance[idx]:.4f}")
    
    def plot_learning_curves(self):
        """Plot learning curves of all selected models plus average"""
        if len(self.selected_models) == 0:
            self._select_best_models()
        
        plt.figure(figsize=(12, 6))
        
        # Convert to numpy arrays for easier computation
        tr_curves = np.array([self.tr_curves[i] for i in self.selected_models])
        va_curves = np.array([self.va_curves[i] for i in self.selected_models])
        
        # Plot individual training curves
        for i, curve in enumerate(tr_curves):
            plt.plot(curve, color='blue', alpha=0.2, linewidth=1)
        
        # Plot individual validation curves
        for i, curve in enumerate(va_curves):
            plt.plot(curve, color='orange', alpha=0.2, linewidth=1)
        
        # Calculate and plot average curves
        avg_tr = tr_curves.mean(axis=0)
        avg_va = va_curves.mean(axis=0)
        
        plt.plot(avg_tr, color='blue', linewidth=3, label='Avg Training')
        plt.plot(avg_va, color='orange', linewidth=3, label='Avg Validation')
        
        plt.title('Learning Curves (Selected Models)')
        plt.xlabel('Epoch')
        plt.ylabel('Negative Log Likelihood')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_variable_importance(self):
        """Plot variable importance based on expert weights"""
        if len(self.selected_models) == 0:
            self._select_best_models()
            
        # Initialize importance accumulators
        trx_importance = {col: 0 for col in TRX_COLS}
        lob_importance = {col: 0 for col in LOB_COLS}
        
        # Process each selected model
        for model_idx in self.selected_models:
            model = self.models[model_idx]
            
            # Get expert weights (absolute values)
            with torch.no_grad():
                # For transaction data expert
                for i, col in enumerate(TRX_COLS):
                    for t in range(self.h):
                        trx_importance[col] += torch.abs(model.ex1.Lm[i*self.h + t]).item()
                
                # For order book data expert
                for i, col in enumerate(LOB_COLS):
                    for t in range(self.h):
                        lob_importance[col] += torch.abs(model.ex2.Lm[i*self.h + t]).item()
        
        # Normalize importance scores
        total_trx = sum(trx_importance.values()) or 1e-8
        total_lob = sum(lob_importance.values()) or 1e-8
        
        trx_importance = {k: v/total_trx for k, v in trx_importance.items()}
        lob_importance = {k: v/total_lob for k, v in lob_importance.items()}
        
        # Create plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Transaction features importance
        trx_df = pd.DataFrame.from_dict(trx_importance, orient='index', columns=['Importance'])
        trx_df.sort_values('Importance', ascending=True).plot(kind='barh', ax=ax1)
        ax1.set_title('Transaction Features Importance')
        ax1.set_xlabel('Normalized Importance Score')
        
        # Order book features importance
        lob_df = pd.DataFrame.from_dict(lob_importance, orient='index', columns=['Importance'])
        lob_df.sort_values('Importance', ascending=True).plot(kind='barh', ax=ax2)
        ax2.set_title('Order Book Features Importance')
        ax2.set_xlabel('Normalized Importance Score')
        
        plt.tight_layout()
        plt.show()
    
    def _fwd(self, model, trx, lob):
        """Forward pass for a single model"""
        mu1, sigma2_1 = model.ex1(trx)
        mu2, sigma2_2 = model.ex2(lob)
        weights = model.gate(trx, lob)
        mu_combined = torch.cat([mu1, mu2], 1)
        sigma2_combined = torch.cat([sigma2_1, sigma2_2], 1)
        return mu_combined, sigma2_combined, weights
    
    def _nll(self, y, mu, sigma2, w):
        """Negative log likelihood for a single model"""
        log_prob = -0.5 * (torch.log(2 * np.pi * sigma2) + (y.unsqueeze(1) - mu)**2 / sigma2)
        return -torch.logsumexp(torch.log(w.clamp(1e-4, 1-1e-4)) + log_prob, 1).mean()
    
    @torch.no_grad()
    def _predict_loader(self, loader):
        """Make predictions using only the selected models"""
        y_log_all, mu_log_all, sigma2_log_all, mv_all = [], [], [], []
        
        for model_idx in self.selected_models:
            model = self.models[model_idx]
            model.eval()
            y_log, mu_log, sigma2_log, mv = [], [], [], []
            
            for trx, lob, y, m in loader:
                trx, lob, y = [t.double().to(DEVICE) for t in (trx, lob, y)]
                mu_k, sigma2_k, w = self._fwd(model, trx, lob)
                
                # Combine predictions from individual experts
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
        
        # Stack predictions from selected models
        y_log = np.stack(y_log_all)  # shape: (n_selected_models, n_samples)
        mu_log = np.stack(mu_log_all)
        sigma2_log = np.stack(sigma2_log_all)
        mv = np.stack(mv_all)
        
        # Compute ensemble statistics
        mu_log_ensemble = mu_log.mean(0)  # average over selected models
        sigma2_log_ensemble = (sigma2_log + mu_log**2).mean(0) - mu_log_ensemble**2
        
        return y_log[0], mu_log_ensemble, sigma2_log_ensemble, mv[0]
    

    def evaluate(self):
        """Evaluate ensemble on test set and plot with datetime x-axis"""
        y_log, mu_log, sigma2_log, mv = self._predict_loader(self.test_dl)

        # Convert to original volume scale with numerical stability
        y_true = np.exp(np.clip(y_log, -700, 700)) * mv
        y_pred = np.exp(np.clip(mu_log, -700, 700)) * mv
        
        # Compute prediction intervals
        sigma2_log_clip = np.clip(sigma2_log, -700, 700)
        mu_log_clip = np.clip(mu_log, -700, 700)
        
        exp_sigma2 = np.exp(sigma2_log_clip)
        exp_2mu_plus_sigma2 = np.exp(2 * mu_log_clip + sigma2_log_clip)
        
        safe_exp = np.where(exp_sigma2 > 1e300, 1e300, exp_sigma2)
        safe_term = (safe_exp - 1) * np.where(exp_2mu_plus_sigma2 > 1e300, 1e300, exp_2mu_plus_sigma2)
        
        sigma = np.sqrt(np.clip(safe_term, 0, 1e300)) * mv
        low, up = y_pred - 0.5*sigma, y_pred + 0.5*sigma

        # Clip extreme values for metrics calculation (unchanged)
        y_true_clip = np.clip(y_true, 0, 1e10)
        y_pred_clip = np.clip(y_pred, 0, 1e10)
        low = np.clip(low, 0, 1e10)
        up = np.clip(up, 0, 1e10)

        # --- NEW: Extract datetime for the test set ---
        # Get the datetime values corresponding to the test predictions
        # Note: We skip the first `h` samples due to the sliding window in WindowDS
        test_datetimes = self.df_test["datetime"].iloc[self.h:].values
        
        # --- Plot with datetime x-axis ---
        plt.figure(figsize=(12, 4))
        plt.fill_between(
            test_datetimes,  # Use datetime for x-axis
            np.clip(low, 0, 5000), 
            np.clip(up, 0, 5000), 
            color="#9ecae1", 
            alpha=0.3, 
            label="95% band"
        )
        plt.plot(test_datetimes, y_true_clip, label="true", lw=0.7, color="#4daf4a")
        plt.plot(test_datetimes, y_pred_clip, label="pred", lw=0.7, color="#e41a1c")
        
        # Format x-axis for better readability
        plt.gcf().autofmt_xdate()  # Auto-rotate datetime labels
        plt.xlabel("Datetime")
        plt.ylabel("Volume")
        plt.title("Test Set Predictions (Time-Dependent)")
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
        print(f"{'NNLL':<10}   | {nnll(y_log, mu_log, sigma2_log):10.4f}")
        
        return y_true_clip, y_pred_clip
        
        
    '''
    def evaluate(self):
        """Evaluate ensemble on test set"""
        y_log, mu_log, sigma2_log, mv = self._predict_loader(self.test_dl)

        # Convert to original volume scale with numerical stability
        y_true = np.exp(np.clip(y_log, -700, 700)) * mv
        y_pred = np.exp(np.clip(mu_log, -700, 700)) * mv
        
        # Compute prediction intervals with numerical stability
        sigma2_log_clip = np.clip(sigma2_log, -700, 700)
        mu_log_clip = np.clip(mu_log, -700, 700)
        
        exp_sigma2 = np.exp(sigma2_log_clip)
        exp_2mu_plus_sigma2 = np.exp(2 * mu_log_clip + sigma2_log_clip)
        
        safe_exp = np.where(exp_sigma2 > 1e300, 1e300, exp_sigma2)
        safe_term = (safe_exp - 1) * np.where(exp_2mu_plus_sigma2 > 1e300, 1e300, exp_2mu_plus_sigma2)
        
        sigma = np.sqrt(np.clip(safe_term, 0, 1e300)) * mv
        low, up = y_pred - 0.5*sigma, y_pred + 0.5*sigma

        # Clip extreme values for metrics calculation
        y_true_clip = np.clip(y_true, 0, 1e10)
        y_pred_clip = np.clip(y_pred, 0, 1e10)
        low = np.clip(low, 0, 1e10)
        up = np.clip(up, 0, 1e10)

        # Print metrics
        print(f"{'Metric':<10} | {'Value':>10}")
        print("-" * 23)
        print(f"{'RMSE':<10}   | {rmse(y_true_clip, y_pred_clip):10.2f}")
        print(f"{'MAE':<10}    | {mae(y_true_clip, y_pred_clip):10.2f}")
        print(f"{'R²':<10}     | {r2_score(y_true_clip, y_pred_clip):10.4f}")
        print(f"{'MAPE (%)':<10}| {mape(y_true_clip, y_pred_clip):10.2f}")
        print(f"{'Coverage':<10}| {np.mean((y_true_clip >= low) & (y_true_clip <= up)) * 100:10.1f} %")
        print(f"{'IW':<10}     | {iw(low, up):10.2f}")
        print(f"{'NNLL':<10}   | {nnll(y_log, mu_log, sigma2_log):10.4f}")
        
        # Plot results
        low_plot = np.zeros_like(y_pred_clip)
        up_plot = np.clip(up, 0, 5000)
        y_pred_plot = np.clip(y_pred, 0, 5000)
        
        plt.figure(figsize=(12, 4))
        plt.fill_between(range(len(y_true_clip)), low_plot, up_plot, 
                        color="#9ecae1", alpha=0.3, label="95% band")
        plt.plot(y_true_clip, label="true", lw=0.7)
        plt.plot(y_pred_plot, label="pred", lw=0.7)
        plt.xlabel("Sample")
        plt.ylabel("Volume")
        plt.legend()
        plt.tight_layout()
        plt.show()

        return y_true_clip, y_pred_clip
    '''
