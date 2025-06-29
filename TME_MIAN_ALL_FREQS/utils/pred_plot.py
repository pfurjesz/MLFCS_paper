import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence, Optional

def plot_volume_forecast(
        y_true   : Sequence[float],
        y_pred   : Sequence[float],
        lower    : Optional[Sequence[float]] = None,
        upper    : Optional[Sequence[float]] = None,
        extra1   : Optional[Sequence[float]] = None,
        extra2   : Optional[Sequence[float]] = None,
        labels   : tuple[str,str,str,str] = ("true", "pred", "extra-1", "extra-2"),
        title    : str  = "Volume forecast",
        band_col : str  = "#9ecae1",
        figsize  : tuple[int,int] = (10,3.5)
    ) -> None:
    """
    Plots y_true & y_pred; optionally a 95 % band and up to two extra curves.

    Parameters
    ----------
    y_true, y_pred   : mandatory 1-D sequences of the same length
    lower, upper     : optional 1-D sequences for confidence limits
    extra1, extra2   : optional extra model outputs / baselines
    labels           : (true, pred, extra1, extra2) names for the legend
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    x      = np.arange(len(y_true))

    if any(len(arr) != len(y_true)
           for arr in (lower, upper, extra1, extra2) if arr is not None):
        raise ValueError("All input series must have the same length")

    plt.figure(figsize=figsize)

    # confidence band (if provided)
    if lower is not None and upper is not None:
        plt.fill_between(x, lower, upper, color=band_col, alpha=.30,
                         label="95 % band")

    # mandatory series
    plt.plot(x, y_true,  label=labels[0], color="#08519c", lw=1.2)
    plt.plot(x, y_pred,  label=labels[1], color="#de2d26", lw=1.2)

    # optional extras
    if extra1 is not None:
        plt.plot(x, extra1, label=labels[2], color="#2ca25f", lw=1)
    if extra2 is not None:
        plt.plot(x, extra2, label=labels[3], color="#756bb1", lw=1)

    plt.xlabel("sample")
    plt.ylabel("volume")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=.3, ls=":")
    plt.tight_layout()
    plt.show()
