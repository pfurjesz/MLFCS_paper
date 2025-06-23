# =====================================================================
# 2.  Utilities  (deseasonalisation + metrics)
# =====================================================================
import pandas as pd, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def deses(df: pd.DataFrame,
          time_col: str = 'datetime',
          volume_col: str = 'total_volume',
          train_size: float = 0.8):
    df  = df.copy()
    df['time'] = df[time_col].dt.time
    train_idx  = int(len(df)*train_size)
    train, rest = df.iloc[:train_idx].copy(), df.iloc[train_idx:].copy()
    mean_vol   = train.groupby('time')[volume_col].mean().replace(0,1e-8)
    for part in (train, rest):
        part['mean_volume'] = part['time'].map(mean_vol).fillna(part[volume_col])
        part['deseasoned_total_volume']  = part[volume_col] / part['mean_volume']
        part['log_deseasoned_total_volume'] = np.log(part['deseasoned_total_volume'].clip(lower=1e-8))
    return pd.concat([train, rest]), rest['mean_volume']

def rmse(y, y_hat): return np.sqrt(mean_squared_error(y, y_hat))
def mae (y, y_hat): return mean_absolute_error(y, y_hat)
def mape(y, y_hat): return np.mean(np.abs((y - y_hat) / np.clip(y,1,None))) * 100

def iw(low, up) -> float:
    """Interval width averaged over the test set (Eq. IW in the paper)."""
    return np.mean(up - low)

def nnll(y_log, mu_log, sigma2_log) -> float:
    """
    Predictive Negative Log-Likelihood normalised by number of points.
    Here computed in log-volume space (exactly as in the paper).
    """
    nll_each = 0.5 * (np.log(2*np.pi*sigma2_log) +
                      (y_log - mu_log)**2 / sigma2_log)
    return nll_each.mean()          # already divided by T


# ---------------------------------------------------------------------
TRX_COLS = ["buy_volume","sell_volume","buy_txn","sell_txn",
            "volume_imbalance","txn_imbalance"]
LOB_COLS = ["ask_volume","bid_volume","spread","lob_volume_imbalance",
            "ask_slope_1","ask_slope_5","ask_slope_10",
            "bid_slope_1","bid_slope_5","bid_slope_10"]

# =====================================================================
# 3.  Dataset   (NO standardisation)
# =====================================================================
class WindowDS(Dataset):
    def __init__(self, df: pd.DataFrame, h: int):
        self.time = df["datetime"].to_numpy()
        self.trx = torch.tensor(df[TRX_COLS].to_numpy(np.float32))
        self.lob = torch.tensor(df[LOB_COLS].to_numpy(np.float32))
        self.y   = torch.tensor(df["log_deseasoned_total_volume"].to_numpy(np.float32))
        self.mv  = torch.tensor(df["mean_volume"].to_numpy(np.float32))
        self.h   = h
    def __len__(self): return len(self.y)-self.h

    def __getitem__(self, i):
        sl = slice(i, i+self.h)
        return self.trx[sl], self.lob[sl], self.y[i+self.h], self.mv[i+self.h]

# =====================================================================
# 4.  Network blocks
# =====================================================================
def _bilinear(x, L, R, b):
    B = x.size(0)
    return (x.permute(0,2,1).reshape(B,-1) @ L).mul(R).sum(1,keepdim=True) + b

class _Expert(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.Lm = nn.Parameter(torch.randn(d*h,1)*.01)
        self.Rm = nn.Parameter(torch.randn(1)*.01)
        self.bm = nn.Parameter(torch.zeros(1))


        self.Lv = nn.Parameter(torch.randn(d*h,1)*.01)
        self.Rv = nn.Parameter(torch.randn(1)*.01)
        self.bv = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mu     = _bilinear(x, self.Lm, self.Rm, self.bm)
        logvar = _bilinear(x, self.Lv, self.Rv, self.bv)
        sigma2 = F.softplus(logvar) + 1e-5
        return mu, sigma2

class _Gate(nn.Module):
    def __init__(self, d_all, h, k=2):
        super().__init__()
        self.L = nn.Parameter(torch.randn(d_all*h, k)*.01)
        self.R = nn.Parameter(torch.randn(1, k)*.01)
        self.b = nn.Parameter(torch.zeros(k))
    def forward(self, *src):
        B = src[0].size(0)
        x = torch.cat([s.permute(0,2,1).reshape(B,-1) for s in src], 1)
        return torch.softmax((x @ self.L).mul(self.R) + self.b, 1)

# =====================================================================
# 5.  TME wrapper
# =====================================================================
class TME:
    def __init__(self, df: pd.DataFrame, cfg: dict):
        self.cfg = cfg
        train_q, val_q = cfg["data_split"]["train_size"], cfg["data_split"]["validation_size"]
        n  = len(df)
        i1 = int(train_q * n); i2 = int((train_q + val_q) * n)
        self.df_train, self.df_val, self.df_test = df.iloc[:i1], df.iloc[i1:i2], df.iloc[i2:]

        h, bs = cfg["model_params"].get("horizon",100), cfg["model_params"]["batch_size"]
        self.train_dl = DataLoader(WindowDS(self.df_train, h), bs, shuffle=True)
        self.val_dl   = DataLoader(WindowDS(self.df_val,   h), bs, shuffle=False)
        self.test_dl  = DataLoader(WindowDS(self.df_test,  h), bs, shuffle=False)
    
        self.net = nn.Module()
        self.net.ex1  = _Expert(len(TRX_COLS), h)
        self.net.ex2  = _Expert(len(LOB_COLS), h)
        self.net.gate = _Gate(len(TRX_COLS)+len(LOB_COLS), h)
        self.net.to(DEVICE).double()

        self.h = h
        self.tr_curve, self.va_curve = [], []

    # ---------- loss
    def _nll(self, y, mu, sigma2, w):
        log_prob = -0.5 * (torch.log(2 * np.pi * sigma2) + (y.unsqueeze(1) - mu)**2 / sigma2)
        return -torch.logsumexp(torch.log(w.clamp(1e-4,1-1e-4)) + log_prob, 1).mean()

    # ---------- forward
    def _fwd(self, trx, lob):
        mu1, sigma2_1 = self.net.ex1(trx)
        mu2, sigma2_2 = self.net.ex2(lob)
        weights       = self.net.gate(trx, lob)
        mu_combined   = torch.cat([mu1, mu2],    1)
        sigma2_combined = torch.cat([sigma2_1, sigma2_2], 1)
        return mu_combined, sigma2_combined, weights

    # ---------- training
    def train(self):
        lr, epochs = self.cfg["model_params"]["learning_rate"], self.cfg["model_params"]["epochs"]
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        best, best_state = np.inf, None
        for ep in range(1, epochs+1):
            self.net.train(); tot = 0.
            for trx, lob, y, _ in self.train_dl:
                trx, lob, y = [t.double().to(DEVICE) for t in (trx, lob, y)]
                opt.zero_grad()
                mu, sigma2, w = self._fwd(trx, lob)
                loss = self._nll(y, mu, sigma2, w)
                loss.backward(); opt.step(); tot += loss.item()
            self.tr_curve.append(tot/len(self.train_dl))

            self.net.eval(); tot = 0.
            with torch.no_grad():
                for trx, lob, y, _ in self.val_dl:
                    trx, lob, y = [t.double().to(DEVICE) for t in (trx, lob, y)]
                    mu, sigma2, w = self._fwd(trx, lob)
                    tot += self._nll(y, mu, sigma2, w).item()
            val = tot/len(self.val_dl); self.va_curve.append(val)
            if val < best:
                best, best_state = val, self.net.state_dict()
            print(f"ep{ep:02d}  train {self.tr_curve[-1]:.4f}  val {val:.4f}")
        self.net.load_state_dict(best_state)

    # ---------- predict loader (returns raw log-space mu/sigma2 + mean_volume)
    @torch.no_grad()
    def _predict_loader(self, loader):
        self.net.eval()
        y_log, mu_log, sigma2_log, mv = [], [], [], []
        for trx, lob, y, m in loader:
            trx, lob, y = [t.double().to(DEVICE) for t in (trx, lob, y)]
            mu_k, sigma2_k, w = self._fwd(trx, lob)
            mu      = (w * mu_k).sum(1)
            sigma2  = (w * sigma2_k).sum(1) + (w * ((mu_k - mu[:, None])**2)).sum(1)
            y_log.append(y.cpu().numpy())
            mu_log.append(mu.cpu().numpy())
            sigma2_log.append(sigma2.cpu().numpy())
            mv.append(m.numpy())
        return map(np.concatenate, (y_log, mu_log, sigma2_log, mv))

    # ---------- evaluate on test
    def evaluate(self):
        y_log, mu_log, sigma2_log, mv = self._predict_loader(self.test_dl)

        y_true = np.exp(y_log) * mv
        y_pred = np.exp(mu_log) * mv
        sigma  = np.sqrt((np.exp(sigma2_log) - 1) * np.exp(2*mu_log + sigma2_log)) * mv
        low, up = y_pred - 1.96*sigma, y_pred + 1.96*sigma

        print(f"{'Metric':<10} | {'Value':>10}")
        print("-" * 23)
        print(f"{'RMSE':<10}   | {rmse(y_true, y_pred):10.2f}")
        print(f"{'MAE':<10}    | {mae(y_true, y_pred):10.2f}")
        print(f"{'R²':<10}     | {r2_score(y_true, y_pred):10.4f}")
        print(f"{'MAPE (%)':<10}| {mape(y_true, y_pred):10.2f}")
        print(f"{'Coverage':<10}| {np.mean((y_true >= low) & (y_true <= up)) * 100:10.1f} %")
        print(f"{'IW':<10}     | {iw(low, up):10.2f}")
        print(f"{'NNLL':<10}   | {nnll(y_log, mu_log, sigma2_log):10.4f}")
        
        # low is just 0 with the same length as y_pred
        low = np.zeros_like(y_pred)  # no lower bound for the band
        up  = np.clip(up, 0, 5000)
        low = np.clip(low, 0, 5000)
        y_pred = np.clip(y_pred, 0, 5000)
        self._plot(y_true, y_pred, low, up, x=self.df_test["datetime"].values[self.h:])

        return(y_true,y_pred)

    # ---------- plotting util

    def _plot(self, y, y_hat, low, up, x=None):
        """
        Plots learning curves and prediction bands.

        Parameters:
            y      : true values (1D array)
            y_hat  : predicted values (1D array)
            low    : lower bound of prediction band
            up     : upper bound of prediction band
            x      : optional x-axis (same length as y); default: range(len(y))
        """
        plt.figure(figsize=(6, 3))
        plt.plot(self.tr_curve, label="train")
        plt.plot(self.va_curve, label="val")
        plt.title("NLL"); plt.legend(); plt.tight_layout(); plt.show()

        if x is None:
            x = range(len(y_hat))

        plt.figure(figsize=(10, 3))
        plt.fill_between(x, low, up, color="#9ecae1", alpha=0.3, label="95 % band")
        plt.plot(x, y,     label="true", lw=0.7)
        plt.plot(x, y_hat, label="pred", lw=0.7)
        if isinstance(x, (pd.Series, np.ndarray)) and np.issubdtype(np.array(x).dtype, np.datetime64):
            plt.xlabel("time")
        else:
            plt.xlabel("sample")
        plt.legend(); plt.tight_layout(); plt.show()
