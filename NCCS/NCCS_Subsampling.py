# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import os
import random
from math import inf
import scipy.io
import joblib
from collections import Counter
from statistics import median, mean

from sklearn.utils import resample
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier


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


def read_into_df(num_days_per_month=3, verbose=False):
    col_names = ['pflag', 'lat', 'lon', 'ts', 'clwp', 'twv', 'PD_10.65', 'PD_89.00', 'PD_166.0'] + [f'tc_{i}' for i in range(13)]
    df = pd.DataFrame(columns=col_names)
    os.chdir("../../../discover/nobackup/jgong/ForSpandan/2017/")
    months = os.listdir(os.getcwd())
    months_dfs = []
    for month in months:
        os.chdir(month)
        files = random.sample(os.listdir(os.getcwd()), num_days_per_month)
        # "THE FORBIDDEN FILE" ;)
        if month == '09':
            while 'colloc_Precipflag_DPR_GMI_20170928.sav' in files:
                files = random.sample(os.listdir(os.getcwd()), num_days_per_month)
        for fl in files:
            data = read_trim(fl)
            if verbose: print(fl)
            PD = make_PDs(data)
            # construct DataFrame
            curr_df = make_df(data, PD)
            # subsample
            curr_df = subsample(curr_df)
            months_dfs.append(curr_df)

        os.chdir("..")
    df = pd.concat(months_dfs, ignore_index=True)
    return df


def prep_data(df):
    num_attribs =  ['lat', 'lon', 'ts', 'clwp', 'twv', 'PD_10.65', 'PD_89.00', 'PD_166.0'] + [f'tc_{i}' for i in range(13)]
    cat_attribs = []


    full_pipeline = ColumnTransformer([
        ("num", StandardScaler(), num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

    return full_pipeline.fit_transform(df)


def random_forest(X, y, vb=False):
    # Random Forest Hyperparameters
    n_estimators = 500
    max_depth = 15
    bootstrap = True
    criterion = 'entropy'
    class_weight = 'balanced_subsample'
    random_state = 42
    n_job = -1
    verbose = 2 if vb else 0

    rfc = RandomForestClassifier(n_estimators=n_estimators, bootstrap=bootstrap, criterion=criterion,
                                 max_depth=max_depth, oob_score=False, verbose=verbose,
                                 class_weight=class_weight, random_state=random_state, n_jobs=n_job)

    rfc.fit(X, y)
    return rfc


def naive_bayes(X, y, vb=False):
    gnb = GaussianNB()
    gnb.fit(X, np.argmax(y, axis=1))
    return gnb


def xgboost_clf(X, y, vb=False):
    xgb = XGBClassifier()
    xgb.fit(X, np.argmax(y, axis=1))
    return xgb


def build_train_model(model_name, X, y, vb=False):
    f_name = {'random_forest': random_forest,
              'naive_bayes': naive_bayes,
              'xgboost_clf': xgboost_clf}

    return f_name[model_name](X, y, vb=vb)


def evaluate(model, X, y):
    y_pred = model.predict(X)
    # y_pred = np.argmax(y_pred, axis=1)
    conf_mat = confusion_matrix(np.argmax(y, axis=1), y_pred)
    pprint(f'\n\n{conf_mat}\n\n')

    print("Accuracy: ", np.trace(conf_mat) / float(np.sum(conf_mat)), end='\n\n')
    y_probs = model.predict_proba(X)
    y_pred_probs = np.column_stack(tuple(y_probs[i][:, 1] for i in range(y.shape[1])))
    auc_roc = roc_auc_score(y, y_pred_probs)
    print("AUC ROC:", auc_roc)


def train(model_name='random_forest', verbose=False):
    # construct dataframe
    df = read_into_df(verbose=verbose)
    if verbose: print(df.head())

    X = df.drop(["pflag"], axis=1)
    y = df[['pflag']]

    # prepare data
    X_prep = prep_data(X)
    y_prep = OneHotEncoder().fit_transform(y).toarray()
    del X
    del y
    if verbose:
        print(f"X: \n{X_prep.shape}")
        print(f"y: \n{y_prep.shape}")
    X_train, X_test, y_train, y_test = train_test_split(X_prep, y_prep, test_size=0.2, random_state=42)
    print(Counter(np.argmax(y_prep, axis=1)))

    # train model
    model = build_train_model(model_name, X_train, y_train, vb=verbose)

    # evaluate model
    evaluate(model, X_test, y_test)

    # save model
    os.chdir("../../../../../../home/sdas11/")
    joblib.dump(model, f'{model_name}.model')


train(model_name='xgboost_clf', verbose=True)
