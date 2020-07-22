# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import os
import joblib
from collections import Counter

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from data_loader import get_data


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
    print(f'\n\n{conf_mat}\n\n')

    print("Accuracy: ", np.trace(conf_mat) / float(np.sum(conf_mat)), end='\n\n')
    y_probs = model.predict_proba(X)
    # print(y_probs.shape, '\n', y_probs)
    # y_pred_probs = np.column_stack(tuple(y_probs[i][:, 1] for i in range(y.shape[1])))
    auc_roc = roc_auc_score(y, y_probs)
    print("AUC ROC:", auc_roc)


def train(model_name='random_forest', verbose=False):
    X_train, X_test, y_train, y_test = get_data()

    # train model
    model = build_train_model(model_name, X_train, y_train, vb=verbose)

    # evaluate model
    evaluate(model, X_test, y_test)

    # save model
    os.chdir("../../../../../../home/sdas11/")
    joblib.dump(model, f'{model_name}.model')


train(model_name='xgboost_clf', verbose=True)
