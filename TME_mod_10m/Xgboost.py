import pandas as pd
import numpy as np
import datetime
from utils import read_txn_data, preprocess_txn_data, create_lob_dataset, merge_txn_and_lob
import xgboost as xgb
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, make_scorer, r2_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV, PredefinedSplit
from matplotlib import pyplot as plt

# train ,val ,test = 0.7, 0.1, 0.2

trx_df = read_txn_data(use_load=False)
lob_df = create_lob_dataset(use_load=False)

trx_df = preprocess_txn_data(trx_df, freq='1min',fill_missing_ts=False)

df_merged = merge_txn_and_lob(trx_df, lob_df)

#recalc deseasoned total volume with mean from training set only
def recalc(df:pd.DataFrame,train_size=0.7)->pd.DataFrame:
    df['time'] = df['datetime'].dt.time
    split = train_test_split(df, train_size=train_size, shuffle=False)
    a = split[0].shape[0]
    train = df.iloc[:a].copy()

    train['mean_volume'] = train.groupby('time')['total_volume'].transform('mean')
    train['deseasoned_total_volume'] = train['total_volume'] / train['mean_volume']
    train['log_deseasoned_total_volume'] = np.log(train['deseasoned_total_volume'])

    rest = df.iloc[a:].copy()

    trange = pd.date_range("00:00:05", "23:59:05", freq='1min').time
    for t in trange:
        if (t in rest['time'].values) and (t in train['time'].values):
            rest.loc[rest.index[rest['time'] == t], 'mean_volume'] = \
            train.loc[train.index[train['time'] == t], 'mean_volume'].iat[0]
        # elif (t in rest['time'].values) and not (t in train['time'].values):
        #     rest.loc[rest.index[rest['time'] == t], 'mean_volume'] = 0
    rest['deseasoned_total_volume'] = rest['total_volume'] / rest['mean_volume']
    rest['log_deseasoned_total_volume'] = np.log(rest['deseasoned_total_volume'])

    return pd.concat([train,rest])

df_merged = recalc(df_merged)

def get_features(df:pd.DataFrame)->np.ndarray:
    #No target to predict for final features
    return df[['buy_volume', 'sell_volume', 'buy_txn', 'sell_txn',
           'volume_imbalance', 'txn_imbalance', 'ask_volume',
           'bid_volume', 'ask_slope_1', 'ask_slope_5', 'ask_slope_10',
           'bid_slope_1', 'bid_slope_5', 'bid_slope_10', 'spread',
           'lob_volume_imbalance', 'slope_imbalance_1', 'slope_imbalance_5',
           'slope_imbalance_10']].values[:-1]

def get_targets(df:pd.DataFrame,log=False)->np.ndarray:
    # No features to predict first target
    return df[['deseasoned_total_volume','total_volume', 'mean_volume','log_deseasoned_total_volume']].values[1:]

dfs_features = get_features(df_merged)
dfs_targets = get_targets(df_merged)

x,x_test,y,y_test = train_test_split(dfs_features,dfs_targets,train_size=0.8,shuffle=False)
x_train,x_val,y_train,y_val = train_test_split(x,y,train_size=7/8,shuffle=False)
# train_idx,_ = train_test_split(range(dfs_features.shape[0]),train_size=0.7,shuffle=False)
# val_idx, _ = train_test_split(_,train_size=2/3,shuffle=False)
split = PredefinedSplit(test_fold=[-1]*len(x_train)+[0]*len(x_val))

# model = xgb.XGBRegressor(eval_metric=['rmse','mae','logloss'])

def custom_error(actual,pred,add=np.array([]),type='sq'):
    #add should be dfs_targets from get_targets
    assert type in ['sq','abs']
    y, y_test = train_test_split( add, train_size=0.8, shuffle=False)
    y_train, y_val = train_test_split(y, train_size=7 / 8, shuffle=False)
    # print('pred shape ')
    # print(pred.shape)
    # print(pred.reshape(-1,1))
    if actual.shape[0] == y_train.shape[0]:
        y_true = y_train[:,1] #y_pred is total volume
        y_pred = pred.reshape(-1,1)*y_train[:,2].reshape(-1,1) #multiply by mean volume
    elif actual.shape[0] == y_val.shape[0]:
        y_true = y_val[:, 1]  # y_pred is total volume
        y_pred = pred.reshape(-1,1)*y_val[:,2].reshape(-1,1) #multiply by mean volume
    elif actual.shape[0] == y_test.shape[0]:
        y_true = y_test[:, 1]  # y_pred is total volume
        y_pred = pred.reshape(-1,1)*y_test[:,2].reshape(-1,1) #multiply by mean volume
    elif actual.shape[0] == y.shape[0]:
        y_true = y[:, 1]  # y_pred is total volume
        y_pred = pred.flatten() * y[:, 2]  # multiply by mean volume

    # print(y_true.shape)
    # print(y_true)
    # print(y_pred.shape)
    # print(y_pred)
    return -root_mean_squared_error(y_true, y_pred) if type == 'sq' else -mean_absolute_error(y_true, y_pred)

rmse = make_scorer(custom_error,greater_is_better=True, add = dfs_targets)
mae = make_scorer(custom_error,greater_is_better=True, add = dfs_targets,type = 'abs')

#mse loss
def custom_obj(actual: np.ndarray,pred: np.ndarray):
    if actual.shape[0] == y_train.shape[0]:
        y_true = y_train[:,1] #y_pred is total volume
        y_pred = pred.flatten()*y_train[:,2] #multiply by mean volume
    elif actual.shape[0] == y_val.shape[0]:
        y_true = y_val[:, 1]  # y_pred is total volume
        y_pred = pred.flatten()*y_val[:,2] #multiply by mean volume
    elif actual.shape[0] == y_test.shape[0]:
        y_true = y_test[:, 1]  # y_pred is total volume
        y_pred = pred.flatten()*y_test[:,2] #multiply by mean volume
    elif actual.shape[0] == y.shape[0]:
        y_true = y[:, 1]  # y_pred is total volume
        y_pred = pred.flatten() * y[:, 2]  # multiply by mean volume

    #loss is 1/2 (x-y)^2
    grad = (y_pred-y_true).reshape(-1,1)
    hess = np.ones_like(grad)
    return grad, hess


# model = xgb.XGBRegressor(objective=custom_obj)
model = xgb.XGBRegressor()
param_grid ={"n_estimators": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
               'subsample': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
               "min_child_weight":[2,3,4,5,6,7,8,9],
              'max_depth': [4,5,6,7,8,9],
              'learning_rate': [0.005, 0.01,0.015,0.02,0.025,0.03,0.035,0.04,0.045,0.05]
             }

# search = RandomizedSearchCV(
#     model, n_iter=500, refit=False, scoring={'rmse':rmse,'mae':mae},param_distributions=param_grid,cv=split
# )

search = RandomizedSearchCV(
    model, n_iter=500, refit=False, scoring="neg_mean_squared_error",param_distributions=param_grid,cv=split
)

search.fit(x,y[:,3])
search_df=pd.DataFrame(search.cv_results_)
# search_df.drop(columns = ['std_test_rmse', 'std_fit_time', 'std_test_mae', 'std_score_time'] ,inplace=True)
search_df.to_csv('xgboost_val/search2-1m.csv')

# search_df[search_df['rank_test_mae'] == 1]
# search_df[search_df['rank_test_rmse'] == 1]
# params=search_df.at[164,'params']

search_df[search_df['rank_test_score'] == 1]
params=search_df.at[314,'params']

# testmodel = xgb.XGBRegressor(objective=custom_obj,**params)
testmodel = xgb.XGBRegressor(**params)
testmodel.fit(x,y[:,3])
# print(root_mean_squared_error(np.maximum(testmodel.predict(x_test),0).reshape(-1,1)*y_test[:,2].reshape(-1,1),y_test[:,1]))
# print(mean_absolute_error(np.maximum(testmodel.predict(x_test),0).reshape(-1,1)*y_test[:,2].reshape(-1,1),y_test[:,1]))
# print(r2_score(np.maximum(testmodel.predict(x_test),0).reshape(-1,1)*y_test[:,2].reshape(-1,1),y_test[:,1]))

print(root_mean_squared_error(np.exp(testmodel.predict(x_test)).reshape(-1,1)*y_test[:,2].reshape(-1,1),y_test[:,1]))
print(mean_absolute_error(np.exp(testmodel.predict(x_test)).reshape(-1,1)*y_test[:,2].reshape(-1,1),y_test[:,1]))
print(r2_score(np.exp(testmodel.predict(x_test)).reshape(-1,1)*y_test[:,2].reshape(-1,1),y_test[:,1]))

#nan for 10 min
# {'subsample': 0.6,
#  'n_estimators': 200,
#  'min_child_weight': 2,
#  'max_depth': 6,
#  'learning_rate': 0.045}

search_df=pd.read_csv('xgboost_val/search2-10m.csv',index_col=0)

plt.figure(figsize=(10, 3.5))
plt.plot(y_test[:,1], label="true", color="#4C72B0", alpha=0.8,
         linewidth=1.0)
plt.plot(np.exp(testmodel.predict(x_test)).reshape(-1,1)*y_test[:,2].reshape(-1,1), label="pred", color="#BD561A", alpha=0.9, linewidth=1.2)
plt.title("Test – total volume")
plt.xlabel("sample")
plt.ylabel("volume")
plt.legend()
plt.tight_layout()
plt.show()