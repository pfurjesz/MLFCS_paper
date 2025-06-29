import pandas as pd
import numpy as np
import os
from datasets import load_dataset


# Configuration settings
DATA_FOLDER = "./data/"
ALLOWED_FREQUENCIES = ['1min', '5min', '10min']
ALLOWED_THRESHOLDS = [1, 5, 10]


def read_txn_data(use_load:bool):
    """
        If use_load is true, load the huggingface dataset (aka start from the beginning).
        If use_load is false, use already preprocessed data (recommended if it already exists)
    """
    if use_load:
        trx_zip_files = [
                os.path.join(DATA_FOLDER, f) for f in os.listdir(DATA_FOLDER) 
                if "trx" in f and f.endswith(".zip")
            ]

        # Ensure that matching files were found
        if not trx_zip_files:
            raise FileNotFoundError("No matching 'trx' ZIP files found in './data/'.")

        # Load dataset using Hugging Face `load_dataset`
        trx_dataset = load_dataset("csv", data_files=trx_zip_files)  # Adjust to "parquet" if needed
        trx_dataset = trx_dataset['train'].to_pandas()
        trx_dataset.rename({col:col.split("'")[1].strip() for col in trx_dataset.columns}, axis=1, inplace=True)
        trx_dataset['datetime'] = pd.to_datetime(trx_dataset['datetime'])
        trx_dataset.to_parquet(f"{DATA_FOLDER}df_full_trx.parquet")
    else:
        try:
            trx_dataset = pd.read_parquet(f"{DATA_FOLDER}df_full_trx.parquet")
            print("trx Data loaded successfully.")
        except FileNotFoundError:
            print(f"Error: The file {DATA_FOLDER}df_full_trx.parquet was not found. Set use_load to true")

    return trx_dataset


def preprocess_txn_data(trx_dataset, freq, second=5, fill_missing_ts=True):
    """
        fill_missing_ts variable should be True,  set it False to check how plots, statistics change if we don't do it
        seconds variable defines from which second the aggregation period starts
    """
    if freq not in ALLOWED_FREQUENCIES:
            raise ValueError(f"Invalid frequency: '{freq}'. Allowed values are '1min', '5min' or '10min'.")

    trx_dataset["datetime"] = trx_dataset["datetime"] - pd.Timedelta(seconds=second)

    # Group by minute and transaction side
    trx_dataset_grouped = trx_dataset.groupby([pd.Grouper(key='datetime', freq=freq, label='right', closed='left'), 'side']).agg(
        volume=('amount', 'sum'),  # Sum of volumes
        txn=('side', 'count')         # Count of transactions
    ).reset_index()

    # Pivot to have separate columns for each transaction type
    trx_dataset_grouped = trx_dataset_grouped.pivot(index='datetime', columns='side')

    # Flatten multi-index columns
    trx_dataset_grouped.columns = [f"{col}_{txn_side}" for txn_side, col in trx_dataset_grouped.columns]

    # Reset index to make time a column again
    trx_dataset_grouped = trx_dataset_grouped.reset_index()
    # Fill missing values with 0
    trx_dataset_grouped = trx_dataset_grouped.fillna(0)

    trx_dataset["datetime"] = trx_dataset["datetime"] + pd.Timedelta(seconds=second)
    trx_dataset_grouped["datetime"] = trx_dataset_grouped["datetime"] + pd.Timedelta(seconds=second)

    if fill_missing_ts:
        # Create a complete range of minutes
        # full_range = pd.DataFrame({'datetime': pd.date_range(start=trx_dataset['datetime'].min().ceil('min') + pd.Timedelta(seconds=second), 
        #                                                 end=trx_dataset['datetime'].max().ceil('min') + pd.Timedelta(seconds=second), 
        #                                                 freq=freq)})

        full_range = pd.DataFrame({'datetime': pd.date_range(start=trx_dataset_grouped['datetime'].min(), 
                                                        end=trx_dataset_grouped['datetime'].max(), 
                                                        freq=freq)})

        # Merge with full time range to ensure all minutes are present
        trx_dataset_grouped = full_range.merge(trx_dataset_grouped, on='datetime', how='left')

        # Fill missing values with 0
        trx_dataset_grouped = trx_dataset_grouped.fillna(0)

    trx_dataset_grouped.rename({col:col.strip() for col in trx_dataset_grouped.columns}, axis=1, inplace=True)
    trx_dataset_grouped['volume_imbalance'] = np.abs(trx_dataset_grouped["buy_volume"] - trx_dataset_grouped['sell_volume'])
    trx_dataset_grouped['txn_imbalance'] = np.abs(trx_dataset_grouped['buy_txn'] - trx_dataset_grouped['sell_txn'])
    trx_dataset_grouped['total_volume'] = trx_dataset_grouped["buy_volume"] + trx_dataset_grouped['sell_volume']

    trx_dataset_grouped['time'] = trx_dataset_grouped['datetime'].dt.time

    trx_dataset_grouped['mean_volume'] = trx_dataset_grouped.groupby('time')['total_volume'].transform('mean')

    trx_dataset_grouped['deseasoned_total_volume'] = trx_dataset_grouped['total_volume'] / trx_dataset_grouped['mean_volume']

    trx_dataset_grouped['log_deseasoned_total_volume'] = np.log(trx_dataset_grouped['deseasoned_total_volume'] + 1e-7)

    trx_dataset_grouped.drop(['time'], axis=1, inplace=True)

    return trx_dataset_grouped


def compute_lob_features(group, thresholds):
    """
        Funtion to be used when grouping LOB data by timestamp
    """
    bids = group[group['type'] == 'b'].sort_values(by='price', ascending=False)  # Highest bid first
    asks = group[group['type'] == 'a'].sort_values(by='price', ascending=True)   # Lowest ask first
    
    if bids.empty or asks.empty:
        return pd.Series({
            'Best Ask': np.nan, 'Best Bid': np.nan, 'ask_volume': np.nan, 'bid_volume': np.nan,
            **{f'ask_slope_{x}': np.nan for x in thresholds},
            **{f'bid_slope_{x}': np.nan for x in thresholds}
        })
    
    best_bid = bids['price'].iloc[0]
    best_ask = asks['price'].iloc[0]
    total_bid_volume = bids['amount'].sum()
    total_ask_volume = asks['amount'].sum()
    
    def find_offset_price(df, total_volume, x):
        """Find price level where cumulative volume reaches x% of total volume."""
        cum_vol = df['amount'].cumsum()
        threshold_volume = total_volume * (x / 100)
        idx = (cum_vol >= threshold_volume).idxmax()
        return df.loc[idx, 'price']
    
    ask_slopes = {}
    bid_slopes = {}
    
    for x in thresholds:
        # Compute p_b_x and delta_x_b
        p_b_x = find_offset_price(bids, total_bid_volume, x)
        delta_x_b = best_bid - p_b_x
        p_a_delta_x_b = best_ask + delta_x_b
        ask_slopes[f'ask_slope_{x}'] = asks[asks['price'] <= p_a_delta_x_b]['amount'].sum()
        
        # Compute p_a_x and delta_x_a
        p_a_x = find_offset_price(asks, total_ask_volume, x, 'ask')
        delta_x_a = best_ask - p_a_x
        p_b_delta_x_a = best_bid + delta_x_a
        bid_slopes[f'bid_slope_{x}'] = bids[bids['price'] >= p_b_delta_x_a]['amount'].sum()
    
    return pd.Series({
        'Best Ask': best_ask, 'Best Bid': best_bid, 'ask_volume': total_ask_volume, 'bid_volume': total_bid_volume,
        **ask_slopes, **bid_slopes
    })


def create_lob_dataset(use_load:bool):
    """
        If use_load is true, load the huggingface dataset (aka start from the beginning).
        If use_load is false, use already preprocessed data (recommended if it already exists)

        !!!!! Because the LOB data is huge, the loading and preprocessing together iteretively file-by-file
    """
    if use_load:
        print("Be ready, It will take ~3 hours.")

        lob_zip_files = [
                os.path.join(DATA_FOLDER, f) for f in os.listdir(DATA_FOLDER) 
                if "ob" in f and f.endswith(".zip")
            ]

        # Ensure that matching files were found
        if not lob_zip_files:
            raise FileNotFoundError("No matching 'ob' ZIP files found in './data/'.")

        df_full_lob = pd.DataFrame()

        for file in lob_zip_files:
            single_file_ob_dataset = [file]
            lob_dataset = load_dataset("csv", data_files=single_file_ob_dataset)

            lob_dataset = lob_dataset['train'].to_pandas()
            lob_dataset['datetime'] = pd.to_datetime(lob_dataset['time'], unit='s', utc=True)
            lob_dataset.drop(['time'], axis=1, inplace = True)
            lob_dataset.rename({col:col.strip() for col in lob_dataset.columns}, axis=1, inplace=True)

            lob_dataset_grouped = lob_dataset.groupby('datetime').apply(lambda group: compute_lob_features(group, ALLOWED_THRESHOLDS), include_groups=False).reset_index()
            df_full_lob = pd.concat([df_full_lob, lob_dataset_grouped], ignore_index=True)
        
        df_full_lob.sort_values('datetime', inplace=True)
        df_full_lob.reset_index(inplace=True)
        df_full_lob['spread'] = df_full_lob['Best Ask'] - df_full_lob['Best Bid']
        df_full_lob['lob_volume_imbalance'] = np.abs(df_full_lob['bid_volume'] - df_full_lob['ask_volume'])
        
        for thresh in ALLOWED_THRESHOLDS:
            df_full_lob[f'slope_imbalance_{thresh}'] = np.abs(df_full_lob[f'ask_slope_{thresh}'] - df_full_lob[f'bid_slope_{thresh}'])
        df_full_lob.drop(['Best Ask', 'Best Bid'], axis=1, inplace=True)

        df_full_lob.to_parquet(f"{DATA_FOLDER}df_full_lob.parquet")

        
    else:
        try:
            df_full_lob = pd.read_parquet(f"{DATA_FOLDER}df_full_lob.parquet")
            print("preprocessed lob Data loaded successfully.")
        except FileNotFoundError:
            print(f"Error: The file {DATA_FOLDER}df_full_lob.parquet was not found. Set use_load to true")
            return 

    return df_full_lob


def merge_txn_and_lob(df_txn, df_lob):
    df_txn.sort_values(by="datetime", inplace=True)
    df_lob.sort_values(by="datetime", inplace=True)
    merged_df = pd.merge_asof(df_txn, df_lob, on="datetime", direction="backward", tolerance = pd.Timedelta("1min"))
    merged_df.dropna(inplace=True)
    return merged_df
