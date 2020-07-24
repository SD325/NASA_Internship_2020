# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import os
import random
from math import inf
import scipy.io
from collections import Counter
from statistics import median, mean

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split



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
               'twv': 'twv',
               'tysfc': 'tysfc'}
    # 2d data
    for data_name, df_name in data_2d.items():
        curr_df[df_name] = np.ravel(data[data_name]).byteswap().newbyteorder()

    # PDs
    for freq in PD_append:
        curr_df[f'PD_{freq}'] = np.ravel(PD[freq]).byteswap().newbyteorder()

    # 3d data
    for idx in range(data['tc'].shape[2]):
        curr_df[f'tc_{idx}'] = np.ravel(data['tc'][:, :, idx]).byteswap().newbyteorder()

    # drop unwanted instances
    curr_df = curr_df[(~np.isnan(curr_df['pflag'])) & (curr_df['ts'] != -99.0) & (~np.isnan(curr_df['twv']))]
    curr_df['pflag'] = curr_df['pflag'].clip(upper=4)
    return curr_df


def subsample(df, balanced=True, verbose=False):
    # print('\t', Counter(df['pflag']))
    if balanced:
        # balanced subsample
        dfs = []
        min_len, max_len = inf, -inf
        lens = []
        for i in range(5):
            dfs.append(df[df['pflag'] == i])
            this_len = len(dfs[i].index)
            lens.append(this_len)
            min_len = min(min_len, this_len)
            max_len = max(max_len, this_len)

        del df
        replace_len = sorted(lens)[1]
        new_dfs = []
        # print(replace_len)
        for i, dfs_i in enumerate(dfs):
            if lens[i] <= replace_len:
                new_dfs.append(dfs_i)
                continue
            # print("before:", dfs_i.shape)
            new_dfs.append(dfs_i.sample(n=replace_len, random_state=42).copy())
            # print("after:", dfs_i.shape)
        del dfs
        df_new = pd.concat(new_dfs)
        del new_dfs
        # print('\t', Counter(df_new['pflag']))
        # random shuffle
        return df_new.sample(frac=1, random_state=42)

    else:
        # random subsample
        return df.sample(frac=0.1, random_state=42)


def read_into_df(num_days_per_month=3, verbose=False, exclude={'colloc_Precipflag_DPR_GMI_20170928.sav'}, testing=False):
    col_names = ['pflag', 'lat', 'lon', 'ts', 'clwp', 'twv', 'tysfc', 'PD_10.65', 'PD_89.00', 'PD_166.0'] + [f'tc_{i}' for i in range(13)]
    # col_names = ['pflag', 'lat', 'lon', 'ts', 'clwp', 'twv', 'PD_10.65', 'PD_89.00', 'PD_166.0'] + [f'tc_{i}' for i in range(13)]
    df = pd.DataFrame(columns=col_names)
    if not testing: os.chdir("../../../discover/nobackup/jgong/ForSpandan/2017/")
    months = os.listdir(os.getcwd())
    months_dfs = []
    used = {'colloc_Precipflag_DPR_GMI_20170928.sav'}
    for month in months:
        os.chdir(month)
        files = random.sample(os.listdir(os.getcwd()), num_days_per_month)
        # "THE FORBIDDEN FILE" ;)
        while any(item in exclude for item in files):
            files = random.sample(os.listdir(os.getcwd()), num_days_per_month)

        for fl in files:
            used.add(fl)
            data = read_trim(fl)
            if verbose: print(fl)
            PD = make_PDs(data)
            # construct DataFrame
            curr_df = make_df(data, PD)
            # subsample
            if not testing: curr_df = subsample(curr_df)
            months_dfs.append(curr_df)

        os.chdir("..")
    df = pd.concat(months_dfs, ignore_index=True)
    return df, used


def prep_data(df):
    num_attribs =  ['lat', 'lon', 'ts', 'clwp', 'twv', 'PD_10.65', 'PD_89.00', 'PD_166.0'] + [f'tc_{i}' for i in range(13)]
    cat_attribs = ['tysfc']


    full_pipeline = ColumnTransformer([
        ("num", StandardScaler(), num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

    return full_pipeline.fit_transform(df), full_pipeline


def get_data(num_days_per_month=3, verbose=False):
    ### TRAIN
    # construct dataframe
    df, used = read_into_df(num_days_per_month=num_days_per_month, verbose=verbose)
    if verbose: print(df.head())

    X = df.drop(["pflag"], axis=1)
    y = df[['pflag']]
    del df

    # prepare data
    X_train, pipeline = prep_data(X)
    y_train = OneHotEncoder().fit_transform(y).toarray()
    del X
    del y
    if verbose:
        print(f"X: \n{X_train.shape}")
        print(f"y: \n{y_train.shape}")
    # X_train, X_test, y_train, y_test = train_test_split(X_prep, y_prep, test_size=0.2, random_state=42)
    print('train: ', Counter(np.argmax(y_train, axis=1)))

    ### TEST
    # construct dataframe
    df, _ = read_into_df(num_days_per_month=1, verbose=verbose, exclude=used, testing=True)
    if verbose: print(df.head())

    X = df.drop(["pflag"], axis=1)
    y = df[['pflag']]
    del df

    # prepare data
    X_test = pipeline.transform(X)  # use train statistics
    y_test = OneHotEncoder().fit_transform(y).toarray()
    del X
    del y
    if verbose:
        print(f"X: \n{X_test.shape}")
        print(f"y: \n{y_test.shape}")

    print('test: ', Counter(np.argmax(y_test, axis=1)))

    # X_train, X_test, y_train, y_test
    return X_train, X_test, y_train, y_test
