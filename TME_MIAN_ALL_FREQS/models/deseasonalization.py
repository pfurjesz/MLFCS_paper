import pandas as pd
import numpy as np

def deses(
    df: pd.DataFrame,
    time_col: str = 'datetime',
    volume_col: str = 'total_volume',
    train_size: float = 0.8
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Removes intraday seasonality from volume data based on time-of-day averages in training data.

    Args:
        df (pd.DataFrame): Input dataframe with datetime and volume columns.
        time_col (str): Name of the datetime column.
        volume_col (str): Name of the volume column.
        train_size (float): Fraction of the data to use for training.

    Returns:
        tuple: Modified dataframe with deseasonalized columns and a Series with removed mean volumes
               (only for the test set, aligned with rest).
    """
    df = df.copy()
    df['time'] = df[time_col].dt.time
    train_idx = int(len(df) * train_size)

    # Split data
    train = df.iloc[:train_idx].copy()
    rest = df.iloc[train_idx:].copy()

    # Compute mean volume per time
    mean_volumes = train.groupby('time')[volume_col].mean()
    mean_volumes.replace(0, 1e-8, inplace=True)

    # Apply to train
    train['mean_volume'] = train['time'].map(mean_volumes)
    train['deseasoned_total_volume'] = train[volume_col] / train['mean_volume']
    train['log_deseasoned_total_volume'] = np.log(train['deseasoned_total_volume'].clip(lower=1e-8))

    # Apply to rest
    rest['mean_volume'] = rest['time'].map(mean_volumes)
    missing_mask = rest['mean_volume'].isna()
    rest.loc[missing_mask, 'mean_volume'] = rest.loc[missing_mask, volume_col]
    rest['mean_volume'].replace(0, 1e-8, inplace=True)
    rest['deseasoned_total_volume'] = rest[volume_col] / rest['mean_volume']
    rest['log_deseasoned_total_volume'] = np.log(rest['deseasoned_total_volume'].clip(lower=1e-8))

    # Return modified dataframe and removed mean volume for rest
    return pd.concat([train, rest]), rest['mean_volume']

