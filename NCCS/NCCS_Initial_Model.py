# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import sys
import os
import scipy.io
import pickle
from collections import Counter
# from imblearn.under_sampling import RandomUnderSampler

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier


def bytesto(bytes, to, bsize=1024):
    a = {'k' : 1, 'm': 2, 'g' : 3, 't' : 4, 'p' : 5, 'e' : 6 }
    r = float(bytes)
    for i in range(a[to]):
        r = r / bsize
    return(r)


def add_to_PD_i(i, freq, ind1, ind2, data):
    PD[i][freq] = data[i]['tc'][:, :, ind1] - data[i]['tc'][:, :, ind2]



os.chdir("../../../discover/nobackup/jgong/ForSpandan/2017/01/")
files = os.listdir(os.getcwd())
# print(len(files))

file_cap = int(sys.argv[1]) if len(sys.argv) == 2 else 6


### READ AND TRIM DATA ########################################################
print("=== Reading and Trimming Data... ===")
data = []
for i, fl in enumerate(files):
    if i >= file_cap:
        continue

    data_i = scipy.io.readsav(fl)
    data_trimmed_i = {}
    for var, arr in data_i.items():
        if arr.ndim == 1:
            data_trimmed_i[var] = arr[1:]
        else:
            data_trimmed_i[var] = arr[1:, 25:196] if arr.ndim == 2 else arr[1:, 25:196, :]

    data.append(data_trimmed_i)

    sz = os.path.getsize(fl)
    str_i = str(i) if i >= 10 else '0' + str(i)
    # print(str_i, "--", fl, " : ", bytesto(sz, 'm'), "MB")

# Polarization Differences (PDs)
PD = [{} for _ in range(len(data))]
for i in range(len(data)):
    add_to_PD_i(i, '10.65', 0, 1, data)
    add_to_PD_i(i, '18.70', 2, 3, data)
    add_to_PD_i(i, '36.50', 5, 6, data)
    add_to_PD_i(i, '89.00', 7, 8, data)
    add_to_PD_i(i, '166.0', 9, 10, data)
    # print(PD[i].keys())

print("=== FINISHED Reading and Trimming Data. ===", end='\n\n')
###############################################################################


### CONSTRUCT DATAFRAME #######################################################
print("=== Preparing Data... ===")
data_dfs = []

PD_freqs_append = ['10.65', '89.00', '166.0']
data_2d = {'latc': 'lat',
           'lonc': 'lon',
           'ts': 'ts',
           'clwp': 'clwp',
           'twv': 'twv'}
for i, data_i in enumerate(data):
        curr_df = pd.DataFrame()
        # 2d data
        for data_name, df_name in data_2d.items():
                curr_df[df_name] = np.ravel(data_i[data_name])

        # PDs
        for freq in PD_freqs_append:
                curr_df[f'PD_{freq}'] = np.ravel(PD[i][freq])

        # 3d data
        for idx in range(data_i['tc'].shape[2]):
                curr_df[f'tc_{idx}'] = np.ravel(data_i['tc'][:, :, idx])

        data_dfs.append(curr_df)
        del curr_df

print("=== Combining DataFrames... ===")
data_df = pd.concat(data_dfs, ignore_index = True)
print("=== FINISHED Combining DataFrames. ===")
del data_dfs

y = pd.DataFrame()
y['pflag'] = np.concatenate(tuple(np.ravel(data[i]['pflag']).byteswap().newbyteorder() for i in range(len(data))))

del data

X_dropped = data_df[(~np.isnan(y['pflag'])) & (data_df['ts'] != -99.0) & (~np.isnan(data_df['twv']))]
y_dropped = y[(~np.isnan(y['pflag'])) & (data_df['ts'] != -99.0) & (~np.isnan(data_df['twv']))]

del data_df
del y


scaler = StandardScaler()
X_prep = scaler.fit_transform(X_dropped)
del X_dropped
ohe = OneHotEncoder()
y_prep = ohe.fit_transform(y_dropped).toarray()
del y_dropped

# should be all zeros
# print("X: ", sum(np.isnan(X_prep)))
# print("y: ", sum(np.isnan(y_prep)))

os.chdir("../../../../../../home/sdas11/")

# with open('X_data.pkl', 'wb') as f:
#     pickle.dump(X_prep, f)
#     del X_prep

# with open('y_data.pkl', 'wb') as f:
#     pickle.dump(y_prep, f)
#     del y_prep

print("=== FINISHED Preparing Data. ===", end='\n\n')
###############################################################################


### SPLIT AND RESAMPLE DATA ###################################################
print("=== Splitting and Resampling Data... ===")
X_train, X_test, y_train, y_test = train_test_split(X_prep, y_prep, test_size=0.2, random_state=42)
del X_prep
del y_prep
# print(f"train: {X_train.shape} {y_train.shape}")
# print(f"test: {X_test.shape} {y_test.shape}")

print("original: ", Counter(np.argmax(y_train, axis=1)))

print("=== FINISHED Splitting and Resampling Data. ===", end= '\n\n')
###############################################################################


### TRAIN MODEL ###############################################################
print("=== Training Model... ===")
# Random Forest Hyperparameters
n_estimators = 5
max_depth = 15
bootstrap = True
criterion = 'entropy'
class_weight = 'balanced'
random_state = 42
n_job = -1



rfc = RandomForestClassifier(n_estimators=n_estimators, bootstrap=bootstrap, criterion=criterion,
                        max_depth=max_depth, oob_score=False,
                        class_weight=class_weight, random_state=random_state, n_jobs=n_job)

rfc.fit(X_train, y_train)
print("=== FINISHED Training Model. ===", end='\n\n')
###############################################################################


### EVALUATE MODEL ############################################################
print("=== Evaluating Model... ===")
y_pred = rfc.predict(X_test)
conf_mat = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
accuracy = np.trace(conf_mat) / float(np.sum(conf_mat))
print(conf_mat)
print("Accuracy: ", accuracy)

print("FINISHED Evaluating Model. ===")
###############################################################################
