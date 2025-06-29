import pandas as pd
import numpy as np
import datetime
import collections

from torch.nn import Softmax

from utils import read_txn_data, preprocess_txn_data, create_lob_dataset, merge_txn_and_lob
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn

from matplotlib import pyplot as plt
import tqdm
import typing

def recalc(df:pd.DataFrame,train_df:pd.DataFrame)->pd.DataFrame:
    df['time'] = df['datetime'].dt.time
    train_size = train_df.shape[0]
    train = train = df.iloc[:train_size].copy()

    train['mean_volume'] = train.groupby('time')['total_volume'].transform('mean')
    train['deseasoned_total_volume'] = train['total_volume'] / train['mean_volume']
    train['log_deseasoned_total_volume'] = np.log(train['deseasoned_total_volume'])

    rest = df.iloc[train_size:].copy()

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

class CustomDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, h: int):
        self.trx = torch.tensor(dataframe[['buy_volume', 'sell_volume', 'buy_txn', 'sell_txn', 'volume_imbalance', 'txn_imbalance']].to_numpy())
        self.lob = torch.tensor(dataframe[['ask_volume', 'bid_volume', 'spread', 'lob_volume_imbalance', 'ask_slope_1', 'ask_slope_5', 'ask_slope_10', 'bid_slope_1', 'bid_slope_5', 'bid_slope_10','slope_imbalance_1', 'slope_imbalance_5',
       'slope_imbalance_10']].to_numpy())
        self.y = torch.tensor(dataframe['log_deseasoned_total_volume'].to_numpy())
        self.h = h

    def __len__(self):
        return len(self.y)-self.h

    def __getitem__(self, idx):
        trx = self.trx[idx:idx+self.h].unfold(0,self.h,1).squeeze()
        lob = self.lob[idx:idx+self.h].unfold(0,self.h,1).squeeze()
        label = self.y[idx+self.h]

        return trx, lob, label
        # return torch.cat((trx, lob),dim=0), label

class latent_dist(nn.Module):
    def __init__(self,h):
        super().__init__()
        self.h=h
        self.R_trx = nn.Linear(self.h, 1,bias=False)
        self.L_trx = nn.Linear(6, 1,bias=True)
        self.R_lob = nn.Linear(self.h, 1,bias=False)
        self.L_lob = nn.Linear(13, 1,bias=True)
        self.soft=nn.Softmax(dim=1)

    def forward(self, trx, lob):
        trx = self.R_trx(trx)
        trx = self.L_trx(torch.permute(trx,(0,2,1)))
        lob = self.R_lob(lob)
        lob = self.L_lob(torch.permute(lob,(0,2,1)))

        return self.soft(torch.cat((trx,lob),dim=1)).squeeze(dim=2)
        # return torch.cat((trx,lob),dim=1).squeeze(dim=2) #return logits

class log_norm(nn.Module):
    def __init__(self,h):
        super().__init__()
        self.h=h
        self.meanR_trx = nn.Linear(self.h, 1,bias=False)
        self.meanL_trx = nn.Linear(6, 1,bias=True)
        self.meanR_lob = nn.Linear(self.h, 1,bias=False)
        self.meanL_lob = nn.Linear(13, 1,bias=True)

        self.varR_trx = nn.Linear(self.h, 1,bias=False)
        self.varL_trx = nn.Linear(6, 1,bias=True)
        self.varR_lob = nn.Linear(self.h, 1,bias=False)
        self.varL_lob = nn.Linear(13, 1,bias=True)


    def forward(self, trx, lob):
        trx_mean = self.meanR_trx(trx)
        trx_mean = self.meanL_trx(torch.permute(trx_mean,(0,2,1)))
        lob_mean = self.meanR_lob(lob)
        lob_mean = self.meanL_lob(torch.permute(lob_mean,(0,2,1)))

        trx_var = self.varR_trx(trx)
        trx_var = self.varL_trx(torch.permute(trx_var,(0,2,1)))
        lob_var = self.varR_lob(lob)
        lob_var = self.varL_lob(torch.permute(lob_var,(0,2,1)))

        #clamp variance in exp, try limit range of var by using tanh
        # return torch.cat((trx_mean,lob_mean),dim=1).squeeze(dim=2), torch.exp(10*torch.tanh(torch.cat((trx_var,lob_var),dim=1)).squeeze(dim=2))
        return torch.cat((trx_mean, lob_mean), dim=1).squeeze(dim=2), torch.exp(
            torch.cat((trx_var, lob_var), dim=1).squeeze(dim=2))

class TME(nn.Module):
    def __init__(self,h):
        super().__init__()
        self.log_norm=log_norm(h)
        self.latent_dist=latent_dist(h)

    def forward(self,trx,lob):
        mean, var = self.log_norm(trx,lob)
        prob = self.latent_dist(trx,lob)
        return mean,var,prob
#input is 20 predictions
class ensemble_probs(nn.Module):
    def __init__(self,n):
        super().__init__()
        self.n=n #number of networks in ensemble. =20
        self.linear = nn.Linear(self.n, self.n,bias=True)
        self.soft=nn.Softmax(dim=1)

    #input 20 predictions of 20 networks
    def forward(self, x):
        x = self.linear(x)
        return self.soft(x)
#input is features
class ensemble_probs(nn.Module):
    def __init__(self,h,n):
        super().__init__()
        self.h=h
        self.n=n #number of models = 20
        self.R = nn.Linear(self.h, self.n,bias=False)
        self.L = nn.Linear(19, 1,bias=True)
        self.soft=nn.Softmax(dim=1)

    #input 20 predictions of 20 networks
    def forward(self, trx, lob):
        #combine both inputs into 1
        x = torch.cat([torch.permute(trx,(1,0,2)),torch.permute(lob,(1,0,2))])
        x = torch.permute(x,(1,0,2)) #number of points, features, window size
        x = self.R(x)
        x = self.L(torch.permute(x,(0,2,1)))

        return self.soft(x.squeeze(dim=2))

def TME_loss(pred,target,eps=1e-6):
    mean,var,prob = pred[0],pred[1],pred[2]
    target = target.unsqueeze(dim=1)
    eps = torch.tensor(eps)
    p1 = torch.exp(-torch.square(target - mean)/(2*var))/(torch.exp(target)*torch.sqrt(var)) #dropped constants
    # return -torch.log(torch.maximum(torch.diag(torch.matmul(p1,torch.permute(prob,(1,0)))),eps)).sum() #sum or mean
    return -torch.log(torch.maximum((p1*prob).sum(1), eps)).mean() #sum or mean

def train_loop(train_dataloader, val_dataloader, model, loss_fn, optimizer, epoch, checkpoint_name:str='training_checkpoint.pth',stop_counter=0, epochs_before_stopping = 10, best_val = float('inf')):
    size = len(train_dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    train_loss = 0
    for batch, (trx,lob, y) in enumerate(train_dataloader):
        # Compute prediction and loss
        pred = model(trx,lob)
        loss = loss_fn(pred, y)
        train_loss += loss.item()
        # print(f"batch number {batch},loss: {loss.item()}")

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # if batch % 100 == 0:
        #     loss, current = loss.item(), batch * batch_size + len(trx)
        #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    if epoch%100 == 0:
        print(f"train loss: {train_loss:>7f}")

    model.eval()
    val_loss = 0
    counter = stop_counter
    with torch.no_grad():
        for trx,lob, y in val_dataloader:
            pred = model(trx,lob)
            val_loss += loss_fn(pred, y).item()

    if val_loss < best_val:
        counter = 0
        best_val = val_loss
        checkpoint = {
            'epoch': epoch+1, #count start from 1
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'loss': val_loss,
        }
        torch.save(checkpoint, checkpoint_name)
        # print('new checkpoint saved')
    else:
        counter += 1
    if epoch%100==0:
        print(f"val loss: {val_loss:>7f}")

    if counter >= epochs_before_stopping:
        return (True, best_val,counter)
    else:
        return (False, best_val,counter)

def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for trx,lob, y in dataloader:
            pred = model(trx,lob)
            test_loss += loss_fn(pred, y).item()

    print(f"Test Error: \n Test Loss: {test_loss:>8f} \n")

#only search over different values of h
def validation_tests(h_search_range=range(3,24*6+3,3),learning_rate = 1e-3, Lambda = 0.1, batch_size = 10918):
    trx_df = read_txn_data(use_load=False)
    lob_df = create_lob_dataset(use_load=False)

    trx_df = preprocess_txn_data(trx_df, freq='10min', fill_missing_ts=False)

    df_merged = merge_txn_and_lob(trx_df, lob_df)
    # split data in train,val, test
    _, test_df = train_test_split(df_merged, train_size=0.8, shuffle=False)
    train_df, val_df = train_test_split(_, train_size=7 / 8, shuffle=False)
    df_merged = recalc(df_merged, train_df) #recalculate deseasonalised vol based on train data only
    _, test_df = train_test_split(df_merged, train_size=0.8, shuffle=False)
    train_df, val_df = train_test_split(_, train_size=7 / 8, shuffle=False)

    #need to already have a dataframe with the right columns
    val_res = pd.read_csv('validation_results_10m/results.csv', index_col=0)
    for h in h_search_range:
        # epochs = 30

        # standardize features
        train_data = CustomDataset((train_df.iloc[:, 1:-1] - train_df.iloc[:, 1:-1].mean()) / train_df.iloc[:, 1:-1].std(),h)
        val_data = CustomDataset((val_df.iloc[:, 1:-1] - train_df.iloc[:, 1:-1].mean()) / train_df.iloc[:, 1:-1].std(), h)
        test_data = CustomDataset((test_df.iloc[:, 1:-1] - train_df.iloc[:, 1:-1].mean()) / train_df.iloc[:, 1:-1].std(), h)

        # train_data = CustomDataset(train_df.iloc[:, 1:-1], h)
        # val_data = CustomDataset(val_df.iloc[:, 1:-1] , h)
        # test_data = CustomDataset(test_df.iloc[:, 1:-1], h)

        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
        val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

        model = TME(h).double()

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=Lambda)

        file_name = f'validation_results_10m\checkpoint{val_res.shape[0]}.pth'

        best_val = float('inf')
        counter = 0
        t=0
        stop = False
        while not stop:
            if t%100==0:
                print(f"Epoch {t + 1}\n-------------------------------")
            stop, best_val, counter = train_loop(train_dataloader, val_dataloader, model, TME_loss, optimizer, t,stop_counter=counter, best_val=best_val,checkpoint_name=file_name)
            if t%100==0:
                test_loop(test_dataloader, model, TME_loss)
            t+=1
            if stop:
                break
        print("Done!")

        #load best validation model for continued training or inference
        loaded_checkpoint = torch.load(file_name, weights_only=True)
        model = TME(h).double()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=Lambda)
        model.load_state_dict(loaded_checkpoint['model_state'])
        optimizer.load_state_dict(loaded_checkpoint['optimizer_state'])

        model.eval()
        size = len(test_dataloader.dataset)
        test_loss = 0
        means = []
        vars = []
        probs = []
        # ys = []

        # check rmse and mae using predictions of raw seasonalised volume. v_t=a_I(t)*y_t
        with torch.no_grad():
            for trx, lob, y in test_dataloader:
                pred1, pred2, pred3 = model(trx, lob)
                means.append(pred1)
                vars.append(pred2)
                probs.append(pred3)
                # ys.append(y)
                test_loss += TME_loss((pred1, pred2, pred3), y).item()
        means = torch.cat((means), 0)
        vars = torch.cat((vars), 0)
        probs = torch.cat((probs), 0)
        # ys = torch.cat((ys), 0)

        pred = torch.exp(train_df.loc[:, 'log_deseasoned_total_volume'].mean() + (means + 0.5 * vars) * train_df.loc[:,
                                                                                                        'log_deseasoned_total_volume'].std())
        pred = (pred * probs).sum(1) * test_df['mean_volume'].iloc[h:].to_numpy()

        rmse = root_mean_squared_error(pred, test_df['total_volume'].iloc[h:])
        mae = mean_absolute_error(pred, test_df['total_volume'].iloc[h:])
        # print(rmse)
        # print(mae)

        new_row = pd.DataFrame({
            'checkpoint': [file_name[23:]],
            'h': [h],
            'lr': [learning_rate],
            'batch_size': [batch_size],
            'lambda': [Lambda],
            'epoch': [loaded_checkpoint['epoch']],
            'val_loss': [loaded_checkpoint['loss']],
            'test_loss': [test_loss],
            'rmse': [rmse],
            'mae': [mae]
        })


        val_res = pd.concat([val_res, new_row], ignore_index=True)
        val_res.to_csv('validation_results_10m/results.csv')

#ran after validation_tests. continue training models switch lr to 1e-4.
# Only improved validation results are saved. Epochs saved is total epochs with lr 1e-4
def continued_validation_tests():
    trx_df = read_txn_data(use_load=False)
    lob_df = create_lob_dataset(use_load=False)

    trx_df = preprocess_txn_data(trx_df, freq='10min', fill_missing_ts=False)

    df_merged = merge_txn_and_lob(trx_df, lob_df)
    # split data in train,val, test
    _, test_df = train_test_split(df_merged, train_size=0.8, shuffle=False)
    train_df, val_df = train_test_split(_, train_size=7 / 8, shuffle=False)
    df_merged = recalc(df_merged, train_df) #recalculate deseasonalised vol based on train data only
    _, test_df = train_test_split(df_merged, train_size=0.8, shuffle=False)
    train_df, val_df = train_test_split(_, train_size=7 / 8, shuffle=False)

    val_res = pd.read_csv('validation_results_10m/results.csv', index_col=0)
    idx = val_res.sort_values('val_loss').index[:] #can restrict to smaller range
    for i in idx:
        if val_res.at[i,'checkpoint'] in [f'checkpoint{k}.pth' for k in range(2,26)]: #in case it is a missing checkpoint
            print(val_res.at[i,'checkpoint'])
            continue
        h = val_res.at[i,'h']
        learning_rate = val_res.at[i,'lr']
        batch_size = val_res.at[i,'batch_size']
        epoch = val_res.at[i,'epoch']
        Lambda = val_res.at[i,'lambda']  # L2 regularisation coefficient

        # standardize features
        train_data = CustomDataset((train_df.iloc[:, 1:-1] - train_df.iloc[:, 1:-1].mean()) / train_df.iloc[:, 1:-1].std(),h)
        val_data = CustomDataset((val_df.iloc[:, 1:-1] - train_df.iloc[:, 1:-1].mean()) / train_df.iloc[:, 1:-1].std(), h)
        test_data = CustomDataset((test_df.iloc[:, 1:-1] - train_df.iloc[:, 1:-1].mean()) / train_df.iloc[:, 1:-1].std(), h)

        # train_data = CustomDataset(train_df.iloc[:, 1:-1], h)
        # val_data = CustomDataset(val_df.iloc[:, 1:-1] , h)
        # test_data = CustomDataset(test_df.iloc[:, 1:-1], h)

        train_dataloader = DataLoader(train_data, batch_size=int(batch_size), shuffle=False)
        val_dataloader = DataLoader(val_data, batch_size=int(batch_size), shuffle=False)
        test_dataloader = DataLoader(test_data, batch_size=int(batch_size), shuffle=False)

        model = TME(h).double()

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=Lambda)

        file_name = f'validation_results_10m\checkpoint{val_res.shape[0]}.pth'

        # load best validation model for continued training or inference
        loaded_checkpoint = torch.load(f'validation_results_10m/{val_res.at[i,"checkpoint"]}', weights_only=True)
        model.load_state_dict(loaded_checkpoint['model_state'])
        optimizer.load_state_dict(loaded_checkpoint['optimizer_state'])
        for g in optimizer.param_groups:
            g['lr'] = 1e-4
        learning_rate = 1e-4

        best_val = val_res.at[i,"val_loss"]
        counter = 0
        t=val_res.at[i,'epoch']
        stop = False
        while not stop:
            if t%100==0:
                print(f"Epoch {t + 1}\n-------------------------------")
            stop, best_val, counter = train_loop(train_dataloader, val_dataloader, model, TME_loss, optimizer, t,stop_counter=counter, best_val=best_val,checkpoint_name=file_name)
            if t%100==0:
                test_loop(test_dataloader, model, TME_loss)
            t+=1
            if stop:
                break
        print("Done!")
        if best_val == val_res.at[i,"val_loss"]: #if continue training has no improvment
            continue
        #load best validation model for continued training or inference
        loaded_checkpoint = torch.load(file_name, weights_only=False) #don't know why false needed
        model = TME(h).double()
        model.load_state_dict(loaded_checkpoint['model_state'])

        model.eval()
        size = len(test_dataloader.dataset)
        test_loss = 0
        means = []
        vars = []
        probs = []
        # ys = []

        # check rmse and mae using predictions of raw seasonalised volume. v_t=a_I(t)*y_t
        with torch.no_grad():
            for trx, lob, y in test_dataloader:
                pred1, pred2, pred3 = model(trx, lob)
                means.append(pred1)
                vars.append(pred2)
                probs.append(pred3)
                # ys.append(y)
                test_loss += TME_loss((pred1, pred2, pred3), y).item()
        means = torch.cat((means), 0)
        vars = torch.cat((vars), 0)
        probs = torch.cat((probs), 0)
        # ys = torch.cat((ys), 0)

        pred = torch.exp(train_df.loc[:, 'log_deseasoned_total_volume'].mean() + (means + 0.5 * vars) * train_df.loc[:,
                                                                                                        'log_deseasoned_total_volume'].std())
        pred = (pred * probs).sum(1) * test_df['mean_volume'].iloc[h:].to_numpy()

        rmse = root_mean_squared_error(pred, test_df['total_volume'].iloc[h:])
        mae = mean_absolute_error(pred, test_df['total_volume'].iloc[h:])
        # print(rmse)
        # print(mae)

        new_row = pd.DataFrame({
            'checkpoint': [file_name[23:]],
            'h': [h],
            'lr': [learning_rate],
            'batch_size': [batch_size],
            'lambda': [Lambda],
            'epoch': [loaded_checkpoint['epoch']],
            'val_loss': [loaded_checkpoint['loss']],
            'test_loss': [test_loss],
            'rmse': [rmse],
            'mae': [mae]
        })


        val_res = pd.concat([val_res, new_row], ignore_index=True)
        val_res.to_csv('validation_results_10m/results.csv')

#ran after continued_validation_tests. continue training models switch lr to 1e-5.
# Only improved validation results are saved. Epochs saved is total epochs with lr 1e-5
def continued_val_tests():
    trx_df = read_txn_data(use_load=False)
    lob_df = create_lob_dataset(use_load=False)

    trx_df = preprocess_txn_data(trx_df, freq='10min', fill_missing_ts=False)

    df_merged = merge_txn_and_lob(trx_df, lob_df)
    # split data in train,val, test
    _, test_df = train_test_split(df_merged, train_size=0.8, shuffle=False)
    train_df, val_df = train_test_split(_, train_size=7 / 8, shuffle=False)
    df_merged = recalc(df_merged, train_df) #recalculate deseasonalised vol based on train data only
    _, test_df = train_test_split(df_merged, train_size=0.8, shuffle=False)
    train_df, val_df = train_test_split(_, train_size=7 / 8, shuffle=False)

    val_res = pd.read_csv('validation_results_10m/results.csv', index_col=0)
    #only keeps rows of h,lambda with best val loss across both lr 1e-3 and 1e-4
    new_val_res = val_res.reset_index().groupby(['h', 'lambda'], as_index=False, sort=False).last()
    idx = new_val_res.sort_values('val_loss').index[:] #can restrict to smaller range
    for i in idx:
        if val_res.at[i,'checkpoint'] in [f'checkpoint{k}.pth' for k in range(2,26)]: #in case it is a missing checkpoint
            print(val_res.at[i,'checkpoint'])
            continue
        h = val_res.at[i,'h']
        learning_rate = val_res.at[i,'lr']
        batch_size = val_res.at[i,'batch_size']
        epoch = val_res.at[i,'epoch']
        Lambda = val_res.at[i,'lambda']  # L2 regularisation coefficient

        # standardize features
        train_data = CustomDataset((train_df.iloc[:, 1:-1] - train_df.iloc[:, 1:-1].mean()) / train_df.iloc[:, 1:-1].std(),h)
        val_data = CustomDataset((val_df.iloc[:, 1:-1] - train_df.iloc[:, 1:-1].mean()) / train_df.iloc[:, 1:-1].std(), h)
        test_data = CustomDataset((test_df.iloc[:, 1:-1] - train_df.iloc[:, 1:-1].mean()) / train_df.iloc[:, 1:-1].std(), h)

        # train_data = CustomDataset(train_df.iloc[:, 1:-1], h)
        # val_data = CustomDataset(val_df.iloc[:, 1:-1] , h)
        # test_data = CustomDataset(test_df.iloc[:, 1:-1], h)

        train_dataloader = DataLoader(train_data, batch_size=int(batch_size), shuffle=False)
        val_dataloader = DataLoader(val_data, batch_size=int(batch_size), shuffle=False)
        test_dataloader = DataLoader(test_data, batch_size=int(batch_size), shuffle=False)

        model = TME(h).double()

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=Lambda)

        file_name = f'validation_results_10m\checkpoint{val_res.shape[0]}.pth'

        # load best validation model for continued training or inference
        loaded_checkpoint = torch.load(f'validation_results_10m/{val_res.at[i,"checkpoint"]}', weights_only=True)
        model.load_state_dict(loaded_checkpoint['model_state'])
        optimizer.load_state_dict(loaded_checkpoint['optimizer_state'])
        for g in optimizer.param_groups:
            g['lr'] = 1e-5
        learning_rate = 1e-5

        best_val = val_res.at[i,"val_loss"]
        counter = 0
        t=val_res.at[i,'epoch']
        stop = False
        while not stop:
            if t%100==0:
                print(f"Epoch {t + 1}\n-------------------------------")
            stop, best_val, counter = train_loop(train_dataloader, val_dataloader, model, TME_loss, optimizer, t,stop_counter=counter, best_val=best_val,checkpoint_name=file_name)
            if t%100==0:
                test_loop(test_dataloader, model, TME_loss)
            t+=1
            if stop:
                break
        print("Done!")
        if best_val == val_res.at[i,"val_loss"]: #if continue training has no improvment
            continue
        #load best validation model for continued training or inference
        loaded_checkpoint = torch.load(file_name, weights_only=False) #don't know why false needed
        model = TME(h).double()
        model.load_state_dict(loaded_checkpoint['model_state'])

        model.eval()
        size = len(test_dataloader.dataset)
        test_loss = 0
        means = []
        vars = []
        probs = []
        # ys = []

        # check rmse and mae using predictions of raw seasonalised volume. v_t=a_I(t)*y_t
        with torch.no_grad():
            for trx, lob, y in test_dataloader:
                pred1, pred2, pred3 = model(trx, lob)
                means.append(pred1)
                vars.append(pred2)
                probs.append(pred3)
                # ys.append(y)
                test_loss += TME_loss((pred1, pred2, pred3), y).item()
        means = torch.cat((means), 0)
        vars = torch.cat((vars), 0)
        probs = torch.cat((probs), 0)
        # ys = torch.cat((ys), 0)

        pred = torch.exp(train_df.loc[:, 'log_deseasoned_total_volume'].mean() + (means + 0.5 * vars) * train_df.loc[:,
                                                                                                        'log_deseasoned_total_volume'].std())
        pred = (pred * probs).sum(1) * test_df['mean_volume'].iloc[h:].to_numpy()

        rmse = root_mean_squared_error(pred, test_df['total_volume'].iloc[h:])
        mae = mean_absolute_error(pred, test_df['total_volume'].iloc[h:])
        # print(rmse)
        # print(mae)

        new_row = pd.DataFrame({
            'checkpoint': [file_name[23:]],
            'h': [h],
            'lr': [learning_rate],
            'batch_size': [batch_size],
            'lambda': [Lambda],
            'epoch': [loaded_checkpoint['epoch']],
            'val_loss': [loaded_checkpoint['loss']],
            'test_loss': [test_loss],
            'rmse': [rmse],
            'mae': [mae]
        })


        val_res = pd.concat([val_res, new_row], ignore_index=True)
        val_res.to_csv('validation_results_10m/results.csv')

def train_ensemble(model_params:dict, df_merged:pd.DataFrame,):
    # split data in train,val, test
    _, test_df = train_test_split(df_merged, train_size=0.8, shuffle=False)
    train_df, val_df = train_test_split(_, train_size=7 / 8, shuffle=False)

    h = model_params['h']
    learning_rate = model_params['lr']
    batch_size = model_params['batch_size']
    Lambda = model_params['lambda']

    train_data = CustomDataset((train_df.iloc[:, 1:-1] - train_df.iloc[:, 1:-1].mean()) / train_df.iloc[:, 1:-1].std(),h)
    val_data = CustomDataset((val_df.iloc[:, 1:-1] - train_df.iloc[:, 1:-1].mean()) / train_df.iloc[:, 1:-1].std(), h)

    train_dataloader = DataLoader(train_data, batch_size=int(batch_size), shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=int(batch_size), shuffle=False)

    for i in range(20):
        model = TME(h).double()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=Lambda)
        file_name = f'ensemble_10m/model{i+1}.pth'
        best_val = float('inf')
        counter = 0
        t = 0
        stop = False
        while not stop:
            if t%100==0:
                print(f"Epoch {t + 1}\n-------------------------------")
            stop, best_val, counter = train_loop(train_dataloader, val_dataloader, model, TME_loss, optimizer, t,stop_counter=counter, best_val=best_val,checkpoint_name=file_name)
            t+=1
            if stop:
                break
        print(f"Trained model {i+1}")

#run to train and save 20 models of TME
def save_ensemble():
    trx_df = read_txn_data(use_load=False)
    lob_df = create_lob_dataset(use_load=False)

    trx_df = preprocess_txn_data(trx_df, freq='10min', fill_missing_ts=False)

    df_merged = merge_txn_and_lob(trx_df, lob_df)
    # split data in train,val, test
    _, test_df = train_test_split(df_merged, train_size=0.8, shuffle=False)
    train_df, val_df = train_test_split(_, train_size=7 / 8, shuffle=False)
    df_merged = recalc(df_merged, train_df) #recalculate deseasonalised vol based on train data only

    val_res = pd.read_csv('validation_results_10m/results.csv', index_col=0)

    train_ensemble(val_res.loc[103],df_merged)
#input is 20 predictions
def train_ensemble_probs_loop(train_x,train_y,val_x,val_y,model,loss_fn, optimizer, epoch, checkpoint_name:str='training_checkpoint.pth',stop_counter=0, epochs_before_stopping = 10, best_val = float('inf')):
    #training starts here after obtaining inputs
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    # Compute prediction and loss
    pred = model(train_x)
    loss = loss_fn((pred*train_x).sum(1), torch.from_numpy(train_y.values)) #MSE loss

    # Backpropagation
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if epoch%100 == 0:
        print(f"train loss: {loss.item():>7f}")

    model.eval()
    counter = stop_counter
    with torch.no_grad():
        pred = model(val_x)
        val_loss = loss_fn((pred*val_x).sum(1), torch.from_numpy(val_y.values)).item()

    if val_loss < best_val:
        counter = 0
        best_val = val_loss
        checkpoint = {
            'epoch': epoch+1, #count start from 1
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'loss': val_loss,
        }
        torch.save(checkpoint, checkpoint_name)
        # print('new checkpoint saved')
    else:
        counter += 1
    if epoch%100==0:
        print(f"val loss: {val_loss:>7f}")

    if counter >= epochs_before_stopping:
        return (True, best_val,counter)
    else:
        return (False, best_val,counter)
#input is features from lob and trx
def train_ensemble_probs_loop(train_dataloader,val_dataloader,train_x,train_y,val_x,val_y,model,loss_fn, optimizer, epoch, checkpoint_name:str='training_checkpoint.pth',stop_counter=0, epochs_before_stopping = 10, best_val = float('inf')):
    #training starts here after obtaining inputs
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    # Compute prediction and loss
    train_loss = 0
    for batch, (trx,lob, y) in enumerate(train_dataloader):
        # Compute prediction and loss
        pred = model(trx,lob)
        loss = loss_fn((pred*train_x).sum(1), y) #MSE loss
        train_loss += loss.item()
        # print(f"batch number {batch},loss: {loss.item()}")

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # if batch % 100 == 0:
        #     loss, current = loss.item(), batch * batch_size + len(trx)
        #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    if epoch%100 == 0:
        print(f"train loss: {train_loss:>7f}")

    model.eval()
    val_loss = 0
    counter = stop_counter
    with torch.no_grad():
        for trx, lob, y in val_dataloader:
            pred = model(trx,lob)
            val_loss += loss_fn((pred * val_x).sum(1), y).item()

    if val_loss < best_val:
        counter = 0
        best_val = val_loss
        checkpoint = {
            'epoch': epoch + 1,  # count start from 1
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'loss': val_loss,
        }
        torch.save(checkpoint, checkpoint_name)
        # print('new checkpoint saved')
    else:
        counter += 1
    if epoch % 100 == 0:
        print(f"val loss: {val_loss:>7f}")

    if counter >= epochs_before_stopping:
        return (True, best_val,counter)
    else:
        return (False, best_val,counter)
#input is 20 predictions
def train_ensemble_probs():
    trx_df = read_txn_data(use_load=False)
    lob_df = create_lob_dataset(use_load=False)

    trx_df = preprocess_txn_data(trx_df, freq='10min', fill_missing_ts=False)

    df_merged = merge_txn_and_lob(trx_df, lob_df)
    # split data in train,val, test
    _, test_df = train_test_split(df_merged, train_size=0.8, shuffle=False)
    train_df, val_df = train_test_split(_, train_size=7 / 8, shuffle=False)
    df_merged = recalc(df_merged, train_df) #recalculate deseasonalised vol based on train data only
    _, test_df = train_test_split(df_merged, train_size=0.8, shuffle=False)
    train_df, val_df = train_test_split(_, train_size=7 / 8, shuffle=False)

    val_res = pd.read_csv('validation_results_10m/results.csv', index_col=0)

    h = val_res.at[103, 'h']
    learning_rate = val_res.at[103, 'lr']
    batch_size = val_res.at[103, 'batch_size']
    Lambda = val_res.at[103, 'lambda']

    # standardize features
    train_data = CustomDataset((train_df.iloc[:, 1:-1] - train_df.iloc[:, 1:-1].mean()) / train_df.iloc[:, 1:-1].std(),h)
    val_data = CustomDataset((val_df.iloc[:, 1:-1] - train_df.iloc[:, 1:-1].mean()) / train_df.iloc[:, 1:-1].std(), h)
    test_data = CustomDataset((test_df.iloc[:, 1:-1] - train_df.iloc[:, 1:-1].mean()) / train_df.iloc[:, 1:-1].std(), h)

    train_predictions = []
    val_predictions = []
    test_predictions = []
    train_dataloader = DataLoader(train_data, batch_size=int(batch_size), shuffle=False)
    val_dataloader = DataLoader(val_data, batch_size=int(batch_size), shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=int(batch_size), shuffle=False)

    IW = torch.ones(()).new_empty((20, len(test_data)))
    for i in range(1,21):
        model = TME(h).double()

        file_name = f'ensemble_10m\model{i}.pth'

        # load best validation model for continued training or inference
        loaded_checkpoint = torch.load(file_name, weights_only=False)
        model.load_state_dict(loaded_checkpoint['model_state'])
        model.eval()
        test_loss = 0
        means = []
        vars = []
        probs = []
        # ys = []

        # check rmse and mae using predictions of raw seasonalised volume. v_t=a_I(t)*y_t
        with torch.no_grad():
            for trx, lob, y in train_dataloader:
                pred1, pred2, pred3 = model(trx, lob)
                means.append(pred1)
                vars.append(pred2)
                probs.append(pred3)
                # ys.append(y)
                test_loss += TME_loss((pred1, pred2, pred3), y).item()
        means = torch.cat((means), 0)
        vars = torch.cat((vars), 0)
        probs = torch.cat((probs), 0)

        target_mean = train_df.loc[:, 'log_deseasoned_total_volume'].mean()
        target_std = train_df.loc[:,'log_deseasoned_total_volume'].std()

        pred = torch.exp(
            target_mean + (means + 0.5 * vars * target_std) * target_std
        )

        pred = (pred * probs).sum(1) * train_df['mean_volume'].iloc[h:].to_numpy()
        train_predictions.append(pred)

        test_loss = 0
        means = []
        vars = []
        probs = []
        # ys = []

        # check rmse and mae using predictions of raw seasonalised volume. v_t=a_I(t)*y_t
        with torch.no_grad():
            for trx, lob, y in val_dataloader:
                pred1, pred2, pred3 = model(trx, lob)
                means.append(pred1)
                vars.append(pred2)
                probs.append(pred3)
                # ys.append(y)
                test_loss += TME_loss((pred1, pred2, pred3), y).item()
        means = torch.cat((means), 0)
        vars = torch.cat((vars), 0)
        probs = torch.cat((probs), 0)

        target_mean = train_df.loc[:, 'log_deseasoned_total_volume'].mean()
        target_std = train_df.loc[:,'log_deseasoned_total_volume'].std()

        pred = torch.exp(
            target_mean + (means + 0.5 * vars * target_std) * target_std
        )

        pred = (pred * probs).sum(1) * val_df['mean_volume'].iloc[h:].to_numpy()
        val_predictions.append(pred)

        test_loss = 0
        means = []
        vars = []
        probs = []
        # ys = []

        # check rmse and mae using predictions of raw seasonalised volume. v_t=a_I(t)*y_t
        with torch.no_grad():
            for trx, lob, y in test_dataloader:
                pred1, pred2, pred3 = model(trx, lob)
                means.append(pred1)
                vars.append(pred2)
                probs.append(pred3)
                # ys.append(y)
                test_loss += TME_loss((pred1, pred2, pred3), y).item()
        means = torch.cat((means), 0)
        vars = torch.cat((vars), 0)
        probs = torch.cat((probs), 0)

        target_mean = train_df.loc[:, 'log_deseasoned_total_volume'].mean()
        target_std = train_df.loc[:,'log_deseasoned_total_volume'].std()

        pred = torch.exp(
            target_mean + (means + 0.5 * vars * target_std) * target_std
        )
        #cannot reorder IW assignment because it uses the previous definition of pred which changes next line
        IW[i-1] = (probs * (cond_var((pred1, pred2, pred3), torch.tensor(test_df["mean_volume"].iloc[h:].to_numpy()), target_mean,
                         target_std) + torch.square(torch.tensor(test_df['mean_volume'].iloc[h:].to_numpy()).unsqueeze(dim=1)*pred))).sum(1)
        pred = (pred * probs).sum(1) * test_df['mean_volume'].iloc[h:].to_numpy()
        test_predictions.append(pred)

    #input for ensemble_probs network
    train_x = torch.stack(train_predictions, dim=1)
    val_x = torch.stack(val_predictions, dim=1)
    test_x = torch.stack(test_predictions, dim=1)
    train_y = train_df['total_volume'].iloc[h:]
    val_y = val_df['total_volume'].iloc[h:]
    test_y = test_df['total_volume'].iloc[h:]

    # for i in range(len(IW)):
    #     IW[i] = IW[i] - test_x.mean(1)

    model = ensemble_probs(20).double()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    file_name = f'ensemble_probs_10m\checkpoint.pth'

    best_val = float('inf')
    counter = 0
    t = 0
    stop = False
    while not stop:
        if t % 100 == 0:
            print(f"Epoch {t + 1}\n-------------------------------")
        stop, best_val, counter = train_ensemble_probs_loop(train_x,train_y,val_x,val_y, model, nn.MSELoss(reduction='mean'), optimizer, t,
                                             stop_counter=counter, best_val=best_val, checkpoint_name=file_name)
        t += 1
        if stop:
            break
    print("Done!")
    # eval on test set

    rmse = root_mean_squared_error((test_x*model(test_x)).sum(1).detach().numpy(), test_y)
    mae = mean_absolute_error((test_x*model(test_x)).sum(1).detach().numpy(), test_y)
    print(f'RMSE = {rmse}')
    print(f'MAE = {mae}')
    # print(f'NNLL = {NNLL(predictions.mean(1),torch.tensor(test_df["total_volume"].iloc[h:].to_numpy())) + Lambda * reg_term/(2*20)}')
    print(f'NNLL = {NNLL((test_x*model(test_x)).sum(1), torch.tensor(test_df["total_volume"].iloc[h:].to_numpy()))}')
    print(f'IW = {torch.sqrt((model(test_x)*torch.permute(IW,[1,0])).sum(1) - torch.square((test_x*model(test_x)).sum(1).detach())).mean()}')

    # plot1()
    # plot2()
#input is features from lob and trx
def train_ensemble_probs():
    trx_df = read_txn_data(use_load=False)
    lob_df = create_lob_dataset(use_load=False)

    trx_df = preprocess_txn_data(trx_df, freq='10min', fill_missing_ts=False)

    df_merged = merge_txn_and_lob(trx_df, lob_df)
    # split data in train,val, test
    _, test_df = train_test_split(df_merged, train_size=0.8, shuffle=False)
    train_df, val_df = train_test_split(_, train_size=7 / 8, shuffle=False)
    df_merged = recalc(df_merged, train_df) #recalculate deseasonalised vol based on train data only
    _, test_df = train_test_split(df_merged, train_size=0.8, shuffle=False)
    train_df, val_df = train_test_split(_, train_size=7 / 8, shuffle=False)

    val_res = pd.read_csv('validation_results_10m/results.csv', index_col=0)

    h = val_res.at[103, 'h']
    learning_rate = val_res.at[103, 'lr']
    batch_size = val_res.at[103, 'batch_size']
    Lambda = val_res.at[103, 'lambda']

    # standardize features
    train_data = CustomDataset((train_df.iloc[:, 1:-1] - train_df.iloc[:, 1:-1].mean()) / train_df.iloc[:, 1:-1].std(),h)
    val_data = CustomDataset((val_df.iloc[:, 1:-1] - train_df.iloc[:, 1:-1].mean()) / train_df.iloc[:, 1:-1].std(), h)
    test_data = CustomDataset((test_df.iloc[:, 1:-1] - train_df.iloc[:, 1:-1].mean()) / train_df.iloc[:, 1:-1].std(), h)

    train_predictions = []
    val_predictions = []
    test_predictions = []
    train_dataloader = DataLoader(train_data, batch_size=int(batch_size), shuffle=False)
    val_dataloader = DataLoader(val_data, batch_size=int(batch_size), shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=int(batch_size), shuffle=False)

    IW = torch.ones(()).new_empty((20, len(test_data)))
    for i in range(1,21):
        model = TME(h).double()

        file_name = f'ensemble_10m\model{i}.pth'

        # load best validation model for continued training or inference
        loaded_checkpoint = torch.load(file_name, weights_only=False)
        model.load_state_dict(loaded_checkpoint['model_state'])
        model.eval()
        test_loss = 0
        means = []
        vars = []
        probs = []
        # ys = []

        # check rmse and mae using predictions of raw seasonalised volume. v_t=a_I(t)*y_t
        with torch.no_grad():
            for trx, lob, y in train_dataloader:
                pred1, pred2, pred3 = model(trx, lob)
                means.append(pred1)
                vars.append(pred2)
                probs.append(pred3)
                # ys.append(y)
                test_loss += TME_loss((pred1, pred2, pred3), y).item()
        means = torch.cat((means), 0)
        vars = torch.cat((vars), 0)
        probs = torch.cat((probs), 0)

        target_mean = train_df.loc[:, 'log_deseasoned_total_volume'].mean()
        target_std = train_df.loc[:,'log_deseasoned_total_volume'].std()

        pred = torch.exp(
            target_mean + (means + 0.5 * vars * target_std) * target_std
        )

        pred = (pred * probs).sum(1) * train_df['mean_volume'].iloc[h:].to_numpy()
        train_predictions.append(pred)

        test_loss = 0
        means = []
        vars = []
        probs = []
        # ys = []

        # check rmse and mae using predictions of raw seasonalised volume. v_t=a_I(t)*y_t
        with torch.no_grad():
            for trx, lob, y in val_dataloader:
                pred1, pred2, pred3 = model(trx, lob)
                means.append(pred1)
                vars.append(pred2)
                probs.append(pred3)
                # ys.append(y)
                test_loss += TME_loss((pred1, pred2, pred3), y).item()
        means = torch.cat((means), 0)
        vars = torch.cat((vars), 0)
        probs = torch.cat((probs), 0)

        target_mean = train_df.loc[:, 'log_deseasoned_total_volume'].mean()
        target_std = train_df.loc[:,'log_deseasoned_total_volume'].std()

        pred = torch.exp(
            target_mean + (means + 0.5 * vars * target_std) * target_std
        )

        pred = (pred * probs).sum(1) * val_df['mean_volume'].iloc[h:].to_numpy()
        val_predictions.append(pred)

        test_loss = 0
        means = []
        vars = []
        probs = []
        # ys = []

        # check rmse and mae using predictions of raw seasonalised volume. v_t=a_I(t)*y_t
        with torch.no_grad():
            for trx, lob, y in test_dataloader:
                pred1, pred2, pred3 = model(trx, lob)
                means.append(pred1)
                vars.append(pred2)
                probs.append(pred3)
                # ys.append(y)
                test_loss += TME_loss((pred1, pred2, pred3), y).item()
        means = torch.cat((means), 0)
        vars = torch.cat((vars), 0)
        probs = torch.cat((probs), 0)

        target_mean = train_df.loc[:, 'log_deseasoned_total_volume'].mean()
        target_std = train_df.loc[:,'log_deseasoned_total_volume'].std()

        pred = torch.exp(
            target_mean + (means + 0.5 * vars * target_std) * target_std
        )
        #cannot reorder IW assignment because it uses the previous definition of pred which changes next line
        IW[i-1] = (probs * (cond_var((pred1, pred2, pred3), torch.tensor(test_df["mean_volume"].iloc[h:].to_numpy()), target_mean,
                         target_std) + torch.square(torch.tensor(test_df['mean_volume'].iloc[h:].to_numpy()).unsqueeze(dim=1)*pred))).sum(1)
        pred = (pred * probs).sum(1) * test_df['mean_volume'].iloc[h:].to_numpy()
        test_predictions.append(pred)

    #input for ensemble_probs network
    train_x = torch.stack(train_predictions, dim=1)
    val_x = torch.stack(val_predictions, dim=1)
    test_x = torch.stack(test_predictions, dim=1)
    train_y = train_df['total_volume'].iloc[h:]
    val_y = val_df['total_volume'].iloc[h:]
    test_y = test_df['total_volume'].iloc[h:]

    # for i in range(len(IW)):
    #     IW[i] = IW[i] - test_x.mean(1)

    model = ensemble_probs(h,20).double()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    file_name = f'ensemble_probs_10m/alt_checkpoint1.pth'

    best_val = float('inf')
    counter = 0
    t = 0
    stop = False
    while not stop:
        if t % 100 == 0:
            print(f"Epoch {t + 1}\n-------------------------------")
        stop, best_val, counter = train_ensemble_probs_loop(train_dataloader,val_dataloader,train_x,train_y,val_x,val_y, model, nn.MSELoss(reduction='mean'), optimizer, t,
                                             stop_counter=counter, best_val=best_val, checkpoint_name=file_name)
        t += 1
        if stop:
            break
    print("Done!")
    # eval on test set
    ens_probs = []
    with torch.no_grad():
        for trx, lob, y in test_dataloader:
            ens_probs.append(model(trx, lob))
    ens_probs = torch.cat(ens_probs, 0)

    rmse = root_mean_squared_error((test_x*ens_probs).sum(1).detach().numpy(), test_y)
    mae = mean_absolute_error((test_x*ens_probs).sum(1).detach().numpy(), test_y)
    print(f'RMSE = {rmse}')
    print(f'MAE = {mae}')
    # print(f'NNLL = {NNLL(predictions.mean(1),torch.tensor(test_df["total_volume"].iloc[h:].to_numpy())) + Lambda * reg_term/(2*20)}')
    print(f'NNLL = {NNLL((test_x*ens_probs).sum(1), torch.tensor(test_df["total_volume"].iloc[h:].to_numpy()))}')
    print(f'IW = {torch.sqrt((ens_probs*torch.permute(IW,[1,0])).sum(1) - torch.square((test_x*ens_probs).sum(1).detach())).mean()}')

    # plot1()
    # plot2()
#input is features from lob and trx with lower learning rate 1e-4
def train_ensemble_probs():
    trx_df = read_txn_data(use_load=False)
    lob_df = create_lob_dataset(use_load=False)

    trx_df = preprocess_txn_data(trx_df, freq='10min', fill_missing_ts=False)

    df_merged = merge_txn_and_lob(trx_df, lob_df)
    # split data in train,val, test
    _, test_df = train_test_split(df_merged, train_size=0.8, shuffle=False)
    train_df, val_df = train_test_split(_, train_size=7 / 8, shuffle=False)
    df_merged = recalc(df_merged, train_df) #recalculate deseasonalised vol based on train data only
    _, test_df = train_test_split(df_merged, train_size=0.8, shuffle=False)
    train_df, val_df = train_test_split(_, train_size=7 / 8, shuffle=False)

    val_res = pd.read_csv('validation_results_10m/results.csv', index_col=0)

    h = val_res.at[103, 'h']
    learning_rate = val_res.at[103, 'lr']
    batch_size = val_res.at[103, 'batch_size']
    Lambda = val_res.at[103, 'lambda']

    # standardize features
    train_data = CustomDataset((train_df.iloc[:, 1:-1] - train_df.iloc[:, 1:-1].mean()) / train_df.iloc[:, 1:-1].std(),h)
    val_data = CustomDataset((val_df.iloc[:, 1:-1] - train_df.iloc[:, 1:-1].mean()) / train_df.iloc[:, 1:-1].std(), h)
    test_data = CustomDataset((test_df.iloc[:, 1:-1] - train_df.iloc[:, 1:-1].mean()) / train_df.iloc[:, 1:-1].std(), h)

    train_predictions = []
    val_predictions = []
    test_predictions = []
    train_dataloader = DataLoader(train_data, batch_size=int(batch_size), shuffle=False)
    val_dataloader = DataLoader(val_data, batch_size=int(batch_size), shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=int(batch_size), shuffle=False)

    IW = torch.ones(()).new_empty((20, len(test_data)))
    for i in range(1,21):
        model = TME(h).double()

        file_name = f'ensemble_10m\model{i}.pth'

        # load best validation model for continued training or inference
        loaded_checkpoint = torch.load(file_name, weights_only=False)
        model.load_state_dict(loaded_checkpoint['model_state'])
        model.eval()
        test_loss = 0
        means = []
        vars = []
        probs = []
        # ys = []

        # check rmse and mae using predictions of raw seasonalised volume. v_t=a_I(t)*y_t
        with torch.no_grad():
            for trx, lob, y in train_dataloader:
                pred1, pred2, pred3 = model(trx, lob)
                means.append(pred1)
                vars.append(pred2)
                probs.append(pred3)
                # ys.append(y)
                test_loss += TME_loss((pred1, pred2, pred3), y).item()
        means = torch.cat((means), 0)
        vars = torch.cat((vars), 0)
        probs = torch.cat((probs), 0)

        target_mean = train_df.loc[:, 'log_deseasoned_total_volume'].mean()
        target_std = train_df.loc[:,'log_deseasoned_total_volume'].std()

        pred = torch.exp(
            target_mean + (means + 0.5 * vars * target_std) * target_std
        )

        pred = (pred * probs).sum(1) * train_df['mean_volume'].iloc[h:].to_numpy()
        train_predictions.append(pred)

        test_loss = 0
        means = []
        vars = []
        probs = []
        # ys = []

        # check rmse and mae using predictions of raw seasonalised volume. v_t=a_I(t)*y_t
        with torch.no_grad():
            for trx, lob, y in val_dataloader:
                pred1, pred2, pred3 = model(trx, lob)
                means.append(pred1)
                vars.append(pred2)
                probs.append(pred3)
                # ys.append(y)
                test_loss += TME_loss((pred1, pred2, pred3), y).item()
        means = torch.cat((means), 0)
        vars = torch.cat((vars), 0)
        probs = torch.cat((probs), 0)

        target_mean = train_df.loc[:, 'log_deseasoned_total_volume'].mean()
        target_std = train_df.loc[:,'log_deseasoned_total_volume'].std()

        pred = torch.exp(
            target_mean + (means + 0.5 * vars * target_std) * target_std
        )

        pred = (pred * probs).sum(1) * val_df['mean_volume'].iloc[h:].to_numpy()
        val_predictions.append(pred)

        test_loss = 0
        means = []
        vars = []
        probs = []
        # ys = []

        # check rmse and mae using predictions of raw seasonalised volume. v_t=a_I(t)*y_t
        with torch.no_grad():
            for trx, lob, y in test_dataloader:
                pred1, pred2, pred3 = model(trx, lob)
                means.append(pred1)
                vars.append(pred2)
                probs.append(pred3)
                # ys.append(y)
                test_loss += TME_loss((pred1, pred2, pred3), y).item()
        means = torch.cat((means), 0)
        vars = torch.cat((vars), 0)
        probs = torch.cat((probs), 0)

        target_mean = train_df.loc[:, 'log_deseasoned_total_volume'].mean()
        target_std = train_df.loc[:,'log_deseasoned_total_volume'].std()

        pred = torch.exp(
            target_mean + (means + 0.5 * vars * target_std) * target_std
        )
        #cannot reorder IW assignment because it uses the previous definition of pred which changes next line
        IW[i-1] = (probs * (cond_var((pred1, pred2, pred3), torch.tensor(test_df["mean_volume"].iloc[h:].to_numpy()), target_mean,
                         target_std) + torch.square(torch.tensor(test_df['mean_volume'].iloc[h:].to_numpy()).unsqueeze(dim=1)*pred))).sum(1)
        pred = (pred * probs).sum(1) * test_df['mean_volume'].iloc[h:].to_numpy()
        test_predictions.append(pred)

    #input for ensemble_probs network
    train_x = torch.stack(train_predictions, dim=1)
    val_x = torch.stack(val_predictions, dim=1)
    test_x = torch.stack(test_predictions, dim=1)
    train_y = train_df['total_volume'].iloc[h:]
    val_y = val_df['total_volume'].iloc[h:]
    test_y = test_df['total_volume'].iloc[h:]

    # for i in range(len(IW)):
    #     IW[i] = IW[i] - test_x.mean(1)

    model = ensemble_probs(h,20).double()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    file_name = f'ensemble_probs_10m/alt_checkpoint1.pth'
    loaded_checkpoint = torch.load(file_name, weights_only=False)
    model = ensemble_probs(24,20).double()
    model.load_state_dict(loaded_checkpoint['model_state'])
    optimizer.load_state_dict(loaded_checkpoint['optimizer_state'])
    for g in optimizer.param_groups:
        g['lr'] = 1e-4
    learning_rate = 1e-4

    file_name = f'ensemble_probs_10m/alt_checkpoint_cont1.pth'

    best_val = 12675.933999
    counter = 0
    t = 0
    stop = False
    while not stop:
        if t % 100 == 0:
            print(f"Epoch {t + 1}\n-------------------------------")
        stop, best_val, counter = train_ensemble_probs_loop(train_dataloader,val_dataloader,train_x,train_y,val_x,val_y, model, nn.MSELoss(reduction='mean'), optimizer, t,
                                             stop_counter=counter, best_val=best_val, checkpoint_name=file_name)
        t += 1
        if stop:
            break
    print("Done!")
    # eval on test set
    ens_probs = []
    with torch.no_grad():
        for trx, lob, y in test_dataloader:
            ens_probs.append(model(trx, lob))
    ens_probs = torch.cat(ens_probs, 0)

    rmse = root_mean_squared_error((test_x*ens_probs).sum(1).detach().numpy(), test_y)
    mae = mean_absolute_error((test_x*ens_probs).sum(1).detach().numpy(), test_y)
    print(f'RMSE = {rmse}')
    print(f'MAE = {mae}')
    # print(f'NNLL = {NNLL(predictions.mean(1),torch.tensor(test_df["total_volume"].iloc[h:].to_numpy())) + Lambda * reg_term/(2*20)}')
    print(f'NNLL = {NNLL((test_x*ens_probs).sum(1), torch.tensor(test_df["total_volume"].iloc[h:].to_numpy()))}')
    print(f'IW = {torch.sqrt((ens_probs*torch.permute(IW,[1,0])).sum(1) - torch.square((test_x*ens_probs).sum(1).detach())).mean()}')

    # plot1()
    # plot2()

def NNLL(pred, target, eps=1e-6):
    mean, var, prob = pred[0], pred[1], pred[2]
    target = target.unsqueeze(dim=1)
    eps = torch.tensor(eps)
    p1 = torch.exp(-torch.square(torch.log(target) - mean) / (2 * var)) / (
                target * torch.sqrt(var))  # dropped constants
    # return -torch.log(torch.maximum(torch.diag(torch.matmul(p1,torch.permute(prob,(1,0)))),eps)).sum() #sum or mean
    return -torch.log(torch.maximum((p1 * prob).sum(1), eps)).mean()  # sum or mean

def cond_var(pred,mean_volume,log_mean,log_std):
    mean, var, prob = pred[0]*log_std+log_mean, pred[1] * torch.square(torch.tensor(log_std)), pred[2]
    res = (torch.exp(var) - 1) * torch.exp(2 * mean + var) * torch.square(mean_volume.unsqueeze(dim=1)) #cond var of raw volume
    return res

def plot1():
    plt.figure(figsize=(10, 3.5))
    plt.fill_between(range(len(predictions.mean(1))), torch.tensor(test_df['mean_volume'].iloc[h:].to_numpy()) * (probs*torch.exp(target_mean + (means - torch.sqrt(vars) * 1.96) * target_std)).sum(1),
                     torch.tensor(test_df['mean_volume'].iloc[h:].to_numpy()) * (probs*torch.exp(target_mean + (means + torch.sqrt(vars) * 1.96) * target_std)).sum(1),color="#4C72B0", alpha=0.15, label="95 % band")
    plt.plot(test_df['total_volume'].iloc[h:].reset_index()['total_volume'], label="true", color="#4C72B0", alpha=.8, linewidth=1.0)
    plt.plot(predictions.mean(1), label="pred", color="#BD561A", alpha=.9, linewidth=1.2)
    plt.title("Test – total volume (±1.96 σ)")
    plt.xlabel("sample");
    plt.ylabel("volume")
    plt.legend();
    plt.tight_layout();
    plt.show()

def plot2():
    plt.figure(figsize=(10, 3.5))
    plt.plot(test_df['total_volume'].iloc[h:].reset_index()['total_volume'], label="true", color="#4C72B0", alpha=0.8, linewidth=1.0)
    plt.plot(predictions.mean(1), label="pred", color="#BD561A", alpha=0.9, linewidth=1.2)
    plt.title("Test – total volume")
    plt.xlabel("sample")
    plt.ylabel("volume")
    plt.legend()
    plt.tight_layout()
    plt.show()

def set_neurips_style():
    plt.rcParams.update({
        "font.family":       "sans-serif",
        "font.size":         12,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.linestyle":    ":",
        "grid.alpha":        0.5,
        "figure.dpi":        120,
        "legend.frameon":    False,
    })

set_neurips_style()

class SWAGScheduler(torch.optim.lr_scheduler.LRScheduler):
    """
    Custom learning rate scheduler that calculates a different learning rate each gradient descent step.
    The default implementation keeps the original learning rate constant, i.e., does nothing.
    You can implement a custom schedule inside calculate_lr,
    and add+store additional attributes in __init__.
    You should not change any other parts of this class.
    """

    def calculate_lr(self, current_epoch: float, previous_lr: float) -> float:
        """
        Calculate the learning rate for the epoch given by current_epoch.
        current_epoch is the fractional epoch of SWA fitting, starting at 0.
        That is, an integer value x indicates the start of epoch (x+1),
        and non-integer values x.y correspond to steps in between epochs (x+1) and (x+2).
        previous_lr is the previous learning rate.

        This method should return a single float: the new learning rate.
        """
        # TODO(2): Implement a custom schedule if desired
        return previous_lr

    # TODO(2): Add and store additional arguments if you decide to implement a custom scheduler
    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            epochs: int,
            steps_per_epoch: int,
    ):
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        super().__init__(optimizer, last_epoch=-1, verbose=False)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, please use `get_last_lr()`.", UserWarning
            )
        return [
            self.calculate_lr(self.last_epoch / self.steps_per_epoch, group["lr"])
            for group in self.optimizer.param_groups
        ]

class SWAGInference(object):

    def __init__(
            self,
            loaders: tuple,
            model_dir: int,
            # TODO(2): optionally add/tweak hyperparameters
            swag_training_epochs: int = 20,
            swag_lr: float = 0.0001,
            swag_update_interval: int = 1,
            max_rank_deviation_matrix: int = 20,
            num_bma_samples: int = 20,
            swag_batch: int = 256
    ):
        """
        :param loaders: tuple of (train,val,test) dataloaders
        :param model_dir: model number 1 to 20 to use for initial weights
        :param swag_training_epochs: Total number of gradient descent epochs for SWAG
        :param swag_lr: Learning rate for SWAG gradient descent
        :param swag_update_interval: Frequency (in epochs) for updating SWAG statistics during gradient descent
        :param max_rank_deviation_matrix: Rank of deviation matrix for full SWAG
        :param num_bma_samples: Number of networks to sample for Bayesian model averaging during prediction
        """

        self.model_dir = model_dir
        self.swag_training_epochs = swag_training_epochs
        self.swag_lr = swag_lr
        self.swag_update_interval = swag_update_interval
        self.max_rank_deviation_matrix = max_rank_deviation_matrix
        self.num_bma_samples = num_bma_samples
        self.swag_batch = swag_batch

        val_res = pd.read_csv('validation_results_10m/results.csv', index_col=0)

        self.h = val_res.at[103, 'h']
        self.learning_rate = val_res.at[103, 'lr']
        self.batch_size = val_res.at[103, 'batch_size']
        self.Lambda = val_res.at[103, 'lambda']

        # Network used to perform SWAG.
        # Note that all operations in this class modify this network IN-PLACE!
        self.network = TME(self.h).double()
        self.file_name = f'ensemble_10m\model{self.model_dir}.pth'

        # load best validation model for continued training or inference
        loaded_checkpoint = torch.load(self.file_name, weights_only=False)
        self.network.load_state_dict(loaded_checkpoint['model_state'])

        # Store training dataset to recalculate batch normalization statistics during SWAG inference
        self.train_dataloader = loaders[0]
        self.val_dataloader = loaders[1]
        self.test_dataloader = loaders[2]

        # SWAG-diagonal
        # TODO(1): create attributes for SWAG-diagonal
        #  Hint: self._create_weight_copy() creates an all-zero copy of the weights
        #  as a dictionary that maps from weight name to values.
        #  Hint: you never need to consider the full vector of weights,
        #  but can always act on per-layer weights (in the format that _create_weight_copy() returns)

        self.SWAG_means = self._create_weight_copy()
        self.SWAG_moments = self._create_weight_copy()
        self.model_number = 0

        # Full SWAG
        # TODO(2): create attributes for SWAG-full
        #  Hint: check collections.deque
        self.deviations = collections.deque([], maxlen=self.max_rank_deviation_matrix)
        self.deviation = self._create_weight_copy()
        self.z_diag2 = None #assign value in sample_paramters()

    def update_swag_statistics(self) -> None:
        """
        Update SWAG statistics with the current weights of self.network.
        """

        # Create a copy of the current network weights
        copied_params = {name: param.detach() for name, param in self.network.named_parameters()}

        # SWAG-diagonal
        for name, param in copied_params.items():
            # TODO(1): update SWAG-diagonal attributes for weight `name` using `copied_params` and `param`
            self.SWAG_means[name] = (self.model_number * self.SWAG_means[name] + param) / (self.model_number + 1)
            self.SWAG_moments[name] = (self.model_number * self.SWAG_moments[name] + param ** 2) / (
                    self.model_number + 1)
            self.deviation[name] = param - self.SWAG_means[name]

        # Full SWAG
        # TODO(2): update full SWAG attributes for weight `name` using `copied_params` and `param`
        self.deviations.append(self.deviation)

    def fit_swag_model(self, loader: torch.utils.data.DataLoader) -> None:
        """
        Fit SWAG on top of the pretrained network self.network.
        This method should perform gradient descent with occasional SWAG updates
        by calling self.update_swag_statistics().
        loader = train_dataloader
        """

        # We use SGD with momentum and weight decay to perform SWA.
        # See the paper on how weight decay corresponds to a type of prior.
        # Feel free to play around with optimization hyperparameters.
        optimizer = torch.optim.SGD(
            self.network.parameters(),
            lr=self.swag_lr,
            momentum=0.9,
            nesterov=False,
            weight_decay=self.Lambda,
        )
        loss_fn = TME_loss
        # TODO(2): Update SWAGScheduler instantiation if you decided to implement a custom schedule.
        #  By default, this scheduler just keeps the initial learning rate given to `optimizer`.
        lr_scheduler = SWAGScheduler(
            optimizer,
            epochs=self.swag_training_epochs,
            steps_per_epoch=len(loader),
        )

        # TODO(1): Perform initialization for SWAG fitting
        for name, param in self.network.named_parameters():
            self.SWAG_means[name] += param.detach()
            self.SWAG_moments[name] += param.detach() ** 2

        self.network.train()
        with tqdm.trange(self.swag_training_epochs, desc="Running gradient descent for SWA") as pbar:
            progress_dict = {}
            for epoch in pbar:
                avg_loss = 0.0
                avg_accuracy = 0.0
                for (trx,lob,y) in loader:
                    optimizer.zero_grad()
                    predictions = self.network(trx,lob)
                    batch_loss = loss_fn(predictions, y)
                    batch_loss.backward()
                    optimizer.step()
                    progress_dict["lr"] = lr_scheduler.get_last_lr()[0]
                    lr_scheduler.step()

                # TODO(1): Implement periodic SWAG updates using the attributes defined in __init__
                if epoch % self.swag_update_interval == 0:
                    self.model_number += 1
                    self.update_swag_statistics()

    def predict_probabilities_swag(self, loader: torch.utils.data.DataLoader,train_df,test_df) -> tuple:
        """
        Perform Bayesian model averaging using your SWAG statistics and predict
        probabilities for all samples in the loader.
        Outputs should be a Nx6 tensor, where N is the number of samples in loader,
        and all rows of the output should sum to 1.
        That is, output row i column j should be your predicted p(y=j | x_i).
        """

        self.network.eval()

        # Perform Bayesian model averaging:
        # Instead of sampling self.num_bma_samples networks (using self.sample_parameters())
        # for each datapoint, you can save time by sampling self.num_bma_samples networks,
        # and perform inference with each network on all samples in loader.
        predictions = []
        IW = torch.ones(()).new_empty((20, len(self.test_dataloader.dataset)))
        for i in tqdm.trange(1,self.num_bma_samples+1, desc="Performing Bayesian model averaging"):
            # TODO(1): Sample new parameters for self.network from the SWAG approximate posterior
            self.sample_parameters()

            # TODO(1): Perform inference for all samples in `loader` using current model sample,
            #  and add the predictions to model_predictions
            means = []
            vars = []
            probs = []
            for (trx,lob,y) in loader:
                pred1, pred2, pred3 = self.network(trx, lob)
                means.append(pred1)
                vars.append(pred2)
                probs.append(pred3)

            means = torch.cat((means), 0)
            vars = torch.cat((vars), 0)
            probs = torch.cat((probs), 0)

            target_mean = train_df.loc[:, 'log_deseasoned_total_volume'].mean()
            target_std = train_df.loc[:, 'log_deseasoned_total_volume'].std()

            pred = torch.exp(
                target_mean + (means + 0.5 * vars * target_std) * target_std
            )
            # cannot reorder IW assignment because it uses the previous definition of pred which changes next line
            IW[i - 1] = (probs * (
                        cond_var((pred1, pred2, pred3), torch.tensor(test_df["mean_volume"].iloc[self.h:].to_numpy()),
                                 target_mean,
                                 target_std) + torch.square(
                    torch.tensor(test_df['mean_volume'].iloc[self.h:].to_numpy()).unsqueeze(dim=1) * pred))).sum(1)
            pred = (pred * probs).sum(1) * test_df['mean_volume'].iloc[self.h:].to_numpy()
            predictions.append(pred)

        predictions = torch.stack(predictions, dim=1)

        return (predictions.mean(1),IW.mean(0))

    def sample_parameters(self) -> None:
        """
        Sample a new network from the approximate SWAG posterior.
        For simplicity, this method directly modifies self.network in-place.
        Hence, after calling this method, self.network corresponds to a new posterior sample.
        """

        # Instead of acting on a full vector of parameters, all operations can be done on per-layer parameters.
        for name, param in self.network.named_parameters():
            # SWAG-diagonal part
            z_diag = torch.randn(param.size())
            # TODO(1): Sample parameter values for SWAG-diagonal
            mean_weights = self.SWAG_means[name]
            std_weights = torch.sqrt(self.SWAG_moments[name] - mean_weights ** 2)
            assert mean_weights.size() == param.size() and std_weights.size() == param.size()

            # Diagonal part
            sampled_weight = mean_weights + std_weights * z_diag

            # Full SWAG part
            # TODO(2): Sample parameter values for full SWAG
            x = 0
            # if self.z_diag2 == None:
            #     self.z_diag2 = torch.randn(self.max_rank_deviation_matrix)
            # for i in range(self.max_rank_deviation_matrix):
            #     x += self.deviations[i][name] * self.z_diag2[i]
            # self.z_diag2 = None #clear again for next sample_paramters() call
            z_diag2 = torch.randn(self.max_rank_deviation_matrix)
            for i in range(self.max_rank_deviation_matrix):
                x += self.deviations[i][name] * z_diag2[i]
            sampled_weight += x/np.sqrt(2*(self.max_rank_deviation_matrix-1))+(1/np.sqrt(2)-1)*(std_weights * z_diag)

        # Modify weight value in-place; directly changing self.network
        param.data = sampled_weight

        # TODO(1): Don't forget to update batch normalization statistics using self._update_batchnorm_statistics()
        #  in the appropriate place!
        self._update_batchnorm_statistics()

    def _create_weight_copy(self) -> typing.Dict[str, torch.Tensor]:
        """Create an all-zero copy of the network weights as a dictionary that maps name -> weight"""
        return {
            name: torch.zeros_like(param, requires_grad=False)
            for name, param in self.network.named_parameters()
        }

    def fit(self,loader: torch.utils.data.DataLoader,) -> None:
        """
        Perform full SWAG fitting procedure.
        If `PRETRAINED_WEIGHTS_FILE` is `True`, this method skips the MAP inference part,
        and uses pretrained weights instead.

        Note that MAP inference can take a very long time.
        You should hence only perform MAP inference yourself after passing the hard baseline
        using the given CNN architecture and pretrained weights.
        """

        # SWAG
        self.fit_swag_model(loader)

    def predict_probabilities(self, loader,df,test_df) -> tuple:
        """
        Predict class probabilities for the given images xs.
        This method returns an NxC float tensor,
        where row i column j corresponds to the probability that y_i is class j.

        This method uses different strategies depending on self.inference_mode.
        """
        self.network = self.network.eval()

        # Create a loader that we can deterministically iterate many times if necessary

        with torch.no_grad():  # save memory by not tracking gradients
            return self.predict_probabilities_swag(loader,df,test_df)

    def _update_batchnorm_statistics(self) -> None:
        """
        Reset and fit batch normalization statistics using the training dataset self.training_dataset.
        We provide this method for you for convenience.
        See the SWAG paper for why this is required.

        Batch normalization usually uses an exponential moving average, controlled by the `momentum` parameter.
        However, we are not training but want the statistics for the full training dataset.
        Hence, setting `momentum` to `None` tracks a cumulative average instead.
        The following code stores original `momentum` values, sets all to `None`,
        and restores the previous hyperparameters after updating batchnorm statistics.
        """

        original_momentum_values = dict()
        for module in self.network.modules():
            # Only need to handle batchnorm modules
            if not isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                continue

            # Store old momentum value before removing it
            original_momentum_values[module] = module.momentum
            module.momentum = None

            # Reset batch normalization statistics
            module.reset_running_stats()

        loader = self.train_dataloader

        self.network.train()
        for (trx,lob,y) in loader:
            self.network(trx,lob)
        self.network.eval()

        # Restore old `momentum` hyperparameter values
        for module, momentum in original_momentum_values.items():
            module.momentum = momentum

def SWAG_main():
    trx_df = read_txn_data(use_load=False)
    lob_df = create_lob_dataset(use_load=False)

    trx_df = preprocess_txn_data(trx_df, freq='10min', fill_missing_ts=False)

    df_merged = merge_txn_and_lob(trx_df, lob_df)
    # split data in train,val, test
    _, test_df = train_test_split(df_merged, train_size=0.8, shuffle=False)
    train_df, val_df = train_test_split(_, train_size=7 / 8, shuffle=False)
    df_merged = recalc(df_merged, train_df) #recalculate deseasonalised vol based on train data only
    _, test_df = train_test_split(df_merged, train_size=0.8, shuffle=False)
    train_df, val_df = train_test_split(_, train_size=7 / 8, shuffle=False)

    val_res = pd.read_csv('validation_results_10m/results.csv', index_col=0)

    h = val_res.at[103, 'h']
    learning_rate = val_res.at[103, 'lr']
    batch_size = val_res.at[103, 'batch_size']
    Lambda = val_res.at[103, 'lambda']

    # standardize features
    train_data = CustomDataset((train_df.iloc[:, 1:-1] - train_df.iloc[:, 1:-1].mean()) / train_df.iloc[:, 1:-1].std(),h)
    val_data = CustomDataset((val_df.iloc[:, 1:-1] - train_df.iloc[:, 1:-1].mean()) / train_df.iloc[:, 1:-1].std(), h)
    test_data = CustomDataset((test_df.iloc[:, 1:-1] - train_df.iloc[:, 1:-1].mean()) / train_df.iloc[:, 1:-1].std(), h)

    train_dataloader = DataLoader(train_data, batch_size=int(batch_size), shuffle=False)
    val_dataloader = DataLoader(val_data, batch_size=int(batch_size), shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=int(batch_size), shuffle=False)

    swag_inference = SWAGInference(
        loaders=(train_dataloader, val_dataloader, test_dataloader),
        model_dir=1,
    )
    swag_dataloader = DataLoader(train_data, batch_size=int(swag_inference.swag_batch), shuffle=True)
    swag_inference.fit(swag_dataloader) #maybe use a loader with smaller batch size

    pred1,pred2 = swag_inference.predict_probabilities(test_dataloader,train_df,test_df)


    rmse = root_mean_squared_error(pred1, test_df['total_volume'].iloc[h:])
    mae = mean_absolute_error(pred1, test_df['total_volume'].iloc[h:])
    print(f'RMSE = {rmse}')
    print(f'MAE = {mae}')
    # print(f'NNLL = {NNLL(predictions.mean(1),torch.tensor(test_df["total_volume"].iloc[h:].to_numpy())) + Lambda * reg_term/(2*20)}')
    print(f'NNLL = {NNLL(pred1, torch.tensor(test_df["total_volume"].iloc[h:].to_numpy()),0)}')
    print(f'IW = {torch.sqrt(pred2 - torch.square(pred1)).mean()}')


if __name__ == "__main__":
    SWAG_main()