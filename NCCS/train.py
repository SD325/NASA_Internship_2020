# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

import sys
import numpy as np
import pandas as pd
import os
import joblib
from collections import Counter

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from data_loader import get_data


def random_forest(X, y, vb=False):
    # Random Forest Hyperparameters
    n_estimators = 500
    max_depth = 20
    bootstrap = True
    criterion = 'entropy'
    class_weight = 'balanced_subsample'
    random_state = 42
    n_job = -1
    verbose = 2 if vb else 0

    rfc = RandomForestClassifier(n_estimators=n_estimators, bootstrap=bootstrap, criterion=criterion,
                                 max_depth=max_depth, oob_score=False, verbose=verbose,
                                 class_weight=class_weight, random_state=random_state, n_jobs=n_job)

    rfc.fit(X, np.argmax(y, axis=1))
    return rfc


def naive_bayes(X, y, vb=False):
    gnb = GaussianNB()
    gnb.fit(X, np.argmax(y, axis=1))
    return gnb


def xgboost_clf(X, y, vb=False):
    xgb = XGBClassifier()
    xgb.fit(X, np.argmax(y, axis=1))
    return xgb


def svm(X, y, vb=False):
    svc = LinearSVC()
    svc.fit(X, np.argmax(y, axis=1))
    return svc


def logistic_regression(X, y, vb=False):
    class_weight = 'balanced'
    n_jobs = -1
    lr = LogisticRegression(class_weight=class_weight, n_jobs=n_jobs)
    lr.fit(X, np.argmax(y, axis=1))
    return lr


def NN(X, y, vb=False):
    hidden_layer_sizes = (100,)
    mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes)
    mlp.fit(X, np.argmax(y, axis=1))
    return mlp


def build_train_model(model_name, X, y, vb=False):
    f_name = {'random_forest': random_forest,
              'naive_bayes': naive_bayes,
              'xgboost_clf': xgboost_clf,
              'svm': svm,
              'logistic_regression': logistic_regression,
              'NN': NN}

    return f_name[model_name](X, y, vb=vb)


def evaluate(model, X, y, model_name='random_forest'):
    y_pred = model.predict(X)
    # if isEnsemble: y_pred = np.argmax(y_pred, axis=1)
    conf_mat = confusion_matrix(np.argmax(y, axis=1), y_pred)
    print(f'\n\n{conf_mat}\n\n')

    print("Accuracy: ", np.trace(conf_mat) / float(np.sum(conf_mat)), end='\n\n')

    if model_name in {'svm'}:
        return

    y_probs = model.predict_proba(X)
    # if model_name == 'random_forest':
    #     y_pred_probs = np.column_stack(tuple(y_probs[i][:, 1] for i in range(y.shape[1])))
    #     auc_roc = roc_auc_score(y, y_pred_probs)
    # else:
    auc_roc = roc_auc_score(y, y_probs)
    print(y_probs.shape, '\n', y_probs)
    print("AUC ROC:", auc_roc)


def train_model(model_name='random_forest', verbose=False):
    X_train, X_test, y_train, y_test = get_data(num_days_per_month=7)

    # train model
    model = build_train_model(model_name, X_train, y_train, vb=verbose)

    # evaluate model
    evaluate(model, X_test, y_test, model_name=model_name)

    # save model
    os.chdir("../../../../../../home/sdas11/")
    joblib.dump(model, f'{model_name}.model')


name = sys.argv[1]
train_model(model_name=name, verbose=True)
