# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

import sys
import numpy as np
import pandas as pd
import os
import random
from math import inf
import scipy.io
from collections import Counter
from statistics import median, mean
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from joblib import dump, load



def read_trim(filename):
    data = scipy.io.readsav(filename)
    data_trimmed = {}
    for var, arr in data.items():
        if arr.ndim == 1:
            data_trimmed[var] = arr[1:]
        elif arr.ndim == 2:
            data_trimmed[var] = arr[1:, 25:196]
        else:
            data_trimmed[var] = arr[1:, 25:196, :]
    del data
    return data_trimmed


def make_PDs(data):
    PD = {'10.65': data['tc'][:, :, 0] - data['tc'][:, :, 1],
          '18.70': data['tc'][:, :, 2] - data['tc'][:, :, 3],
          '36.50': data['tc'][:, :, 5] - data['tc'][:, :, 6],
          '89.00': data['tc'][:, :, 7] - data['tc'][:, :, 8],
          '166.0': data['tc'][:, :, 9] - data['tc'][:, :, 10]}
    return PD


def make_df(data, PD):
    curr_df = pd.DataFrame()
    PD_append = ['10.65', '89.00', '166.0']
    data_2d = {'pflag': 'pflag',
               'latc': 'lat',
               'lonc': 'lon',
               'ts': 'ts',
               'clwp': 'clwp',
               'twv': 'twv'} # ,
               # 'tysfc': 'tysfc'}
    # 2d data
    for data_name, df_name in data_2d.items():
        curr_df[df_name] = np.ravel(data[data_name]).byteswap().newbyteorder()

    # PDs
    for freq in PD_append:
        curr_df[f'PD_{freq}'] = np.ravel(PD[freq]).byteswap().newbyteorder()

    # 3d data
    for idx in range(data['tc'].shape[2]):
        curr_df[f'tc_{idx}'] = np.ravel(data['tc'][:, :, idx]).byteswap().newbyteorder()
        curr_df[f'emis_{idx}'] = np.ravel(data['emis'][:, :, idx]).byteswap().newbyteorder()

    # drop unwanted instances
    curr_df = curr_df[(curr_df['ts'] != -99.0) & (~np.isnan(curr_df['twv']))]
    curr_df['pflag'] = curr_df['pflag'].clip(upper=4)
    return curr_df


def filter_df(df, eps=5.0):
    ltb = [14.2, 18.0]  # latitude bounds
    lnb = [-5.0, -2.1]  # longitude bounds
    ltb[0] -= eps
    ltb[1] += eps
    lnb[0] -= eps
    lnb[1] += eps
    return df[(ltb[0] <= df['lat']) & (df['lat'] <= ltb[1]) & (lnb[0] <= df['lon']) & (df['lon'] <= lnb[1])]


def read_squall_line():
    # col_names = ['pflag', 'lat', 'lon', 'ts', 'clwp', 'twv', 'tysfc', 'PD_10.65', 'PD_89.00', 'PD_166.0'] + [f'tc_{i}' for i in range(13)]
    # df = pd.DataFrame(columns=col_names)
    os.chdir("../../../discover/nobackup/jgong/ForSpandan/2017/06/")

    fl = 'colloc_Precipflag_DPR_GMI_20170606.sav'
    data = read_trim(fl)
    PD = make_PDs(data)

    os.chdir("..")
    # construct DataFrame
    df = make_df(data, PD)
    return filter_df(df)



def get_data(verbose=False):
    # get dataframe
    df = read_squall_line()
    # print(df.tysfc.unique())
    X = df.drop(["pflag"], axis=1)
    y = df[['pflag']]
    del df

    os.chdir("../../../../../../home/sdas11/")
    pipeline = load('full_pipeline.bin')

    # prepare data
    X_sq = pipeline.transform(X)  # use train statistics
    y_sq = y.copy()
    # del X
    del y
    # print(f"X: \n{X_sq.shape}")
    # print(f"y: \n{y_sq.shape}")

    # print('squall line: \n', y_sq['pflag'].value_counts())

    return X, X_sq, y_sq


def get_counts(model_name='random_forest', threshold=0.8, use_thld=False):
    X_df, X_sq, y_sq = get_data()
    lat_unf, lon_unf, counts_unf = [], [], []
    lat, lon, counts = [], [], []
    # print("X: ", sum(np.isnan(X_sq)))  # should be all zeros

    model = load(f'{model_name}.model')
    # y_pred = np.argmax(model.predict(X_sq), axis=1)
    y_pred = model.predict(X_sq)
    # print(X_df.iloc[10, 0])
    y_probs = model.predict_proba(X_sq)
    y_pred_probs = np.max(y_probs, axis=1)

    row_set = set()
    for r, _ in np.ndindex(X_sq.shape):
        if r in row_set:
            continue
        row_set.add(r)
        if not np.isnan(y_sq.iloc[r, 0]):
            lat_unf.append(X_df.iloc[r, 0])
            lon_unf.append(X_df.iloc[r, 1])
            counts_unf.append(y_sq.iloc[r, 0])
        if use_thld and y_pred_probs[r] < threshold:
            continue
        lat.append(X_df.iloc[r, 0])
        lon.append(X_df.iloc[r, 1])
        counts.append(y_pred[r].astype(float))
    return lat, lon, counts, lat_unf, lon_unf, counts_unf


def plot_squall_line(model_name='random_forest'):
    lat, lon, counts, lat_unf, lon_unf, counts_unf = get_counts(model_name=model_name)
    '''print(len(lat))
    print(len(lon))
    print(len(counts))'''
    title = "Precip. Type Filled"
    plt.scatter(lon, lat, c=counts, cmap=plt.get_cmap("jet"), vmin=0, vmax=4)
    plt.colorbar()
    plt.title(title)
    plt.show()

    '''print(len(lat_unf))
    print(len(lon_unf))
    print(len(counts_unf))'''
    title = "Precip. Type Unfilled"
    plt.scatter(lon_unf, lat_unf, c=counts_unf, cmap=plt.get_cmap("jet"), vmin=0, vmax=4)
    plt.colorbar()
    plt.title(title)
    plt.show()



name = sys.argv[1]
plot_squall_line(model_name=name)
