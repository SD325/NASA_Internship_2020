# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import os
import random
import scipy.io
from sklearn.utils import resample


def read_trim(filename):
    data = scipy.io.readsav(filename)
    data_trimmed = {}
    for var, arr in data.items():
        if arr.ndim == 1:
            data_trimmed[var] = arr[1:]
        else:
            data_trimmed[var] = arr[1:, 25:196] if arr.ndim == 2 else arr[1:, 25:196, :]
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
               'twv': 'twv'}
    # 2d data
    for data_name, df_name in data_2d.items():
        curr_df[df_name] = np.ravel(data[data_name])

    # PDs
    for freq in PD_append:
        curr_df[f'PD_{freq}'] = np.ravel(PD[freq])

    # 3d data
    for idx in range(data['tc'].shape[2]):
        curr_df[f'tc_{idx}'] = np.ravel(data['tc'][:, :, idx])

    # drop unwanted instances
    curr_df = curr_df[(~np.isnan(curr_df['pflag'])) & (curr_df['ts'] != -99.0) & (~np.isnan(curr_df['twv']))]

    return curr_df


def read_into_df(num_days_per_month=3):
    df = pd.DataFrame()
    os.chdir("../../../discover/nobackup/jgong/ForSpandan/2017/")
    months = os.listdir(os.getcwd())
    for month in months:
        os.chdir(month)
        files = random.sample(os.listdir(os.getcwd()), num_days_per_month)
        for fl in files:
            data = read_trim(fl)
            PD = make_PDs(data)
            # construct DataFrame
            curr_df = make_df(data, PD)

        os.chdir("..")

    return df


