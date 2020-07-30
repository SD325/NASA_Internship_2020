# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

import sys
import numpy as np
import pandas as pd
import os
import joblib
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from data_loader import get_data


def random_forest(X, y, vb=False):
    # Random Forest Hyperparameters
    n_estimators = 50
    max_depth = 20
    bootstrap = True
    criterion = 'entropy'
    class_weight = 'balanced_subsample'
    random_state = 42
    n_job = -1
    verbose = 0

    rfc = RandomForestClassifier(n_estimators=n_estimators, bootstrap=bootstrap, criterion=criterion,
                                 max_depth=max_depth, oob_score=False, verbose=verbose,
                                 class_weight=class_weight, random_state=random_state, n_jobs=n_job)

    rfc.fit(X, np.argmax(y, axis=1))
    rfc.verbose = False
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


def pred_prob_hist(y, y_pred, y_probs, model_name='random_forest', hist=False):
    # Correct ONLY
    sns.distplot(y_probs[:, 0][(y==0) & (y_pred==0)], hist=hist, label='no precipitation')
    sns.distplot(y_probs[:, 1][(y==1) & (y_pred==1)], hist=hist, label='stratiform')
    sns.distplot(y_probs[:, 2][(y==2) & (y_pred==2)], hist=hist, label='convective')
    sns.distplot(y_probs[:, 3][(y==3) & (y_pred==3)], hist=hist, label='other')
    sns.distplot(y_probs[:, 4][(y==4) & (y_pred==4)], hist=hist, label='mixed').set_title('Correct Probabilities')
    plt.show()

    # Incorrect ONLY
    sns.distplot(y_probs[:, 0][(y==0) & (y_pred!=0)], hist=hist, label='no precipitation')
    sns.distplot(y_probs[:, 1][(y==1) & (y_pred!=1)], hist=hist, label='stratiform')
    sns.distplot(y_probs[:, 2][(y==2) & (y_pred!=2)], hist=hist, label='convective')
    sns.distplot(y_probs[:, 3][(y==3) & (y_pred!=3)], hist=hist, label='other')
    sns.distplot(y_probs[:, 4][(y==4) & (y_pred!=4)], hist=hist, label='mixed').set_title('Incorrect Probabilities')
    plt.show()



def evaluate(model, X, y, model_name='random_forest', plot_probs=False):
    y_argmax = np.argmax(y, axis=1)
    y_pred = model.predict(X)

    conf_mat = confusion_matrix(y_argmax, y_pred)
    print(f'Confusion Matrix: \n\n{conf_mat}\n\n')

    print("Accuracy: ", np.trace(conf_mat) / float(np.sum(conf_mat)), end='\n\n')

    print(f'Classification Report: \n{classification_report(y_argmax, y_pred)}', end='\n\n')

    if model_name in {'svm'}:
        return

    y_probs = model.predict_proba(X)
    auc_roc = roc_auc_score(y, y_probs)
    print("AUC ROC: ", auc_roc, end='\n\n')

    if plot_probs:
        pred_prob_hist(y_argmax, y_pred, y_probs, model_name=model_name, hist=False)


def feat_imp(model, model_name='random_forest', show_plot=True):
    # Feature Importances
    if hasattr(model, "feature_importances_"):
        feature_names = np.array(['lat', 'lon', 'ts', 'clwp', 'twv', 'PD_10.65', 'PD_89.00', 'PD_166.0'] + [f'tc_{i}' for i in range(13)] + [f'emis_{i}' for i in range(13)])
        # feature_names = np.array(['lat', 'lon', 'ts', 'clwp', 'twv'] + [f'tysfc_{i}' for i in range(13)] + ['PD_10.65', 'PD_89.00', 'PD_166.0'] + [f'tc_{i}' for i in range(13)] + [f'emis_{i}' for i in range(13)])

        fi = model.feature_importances_
        importance_sorted_idx = np.argsort(fi)
        print(feature_names[importance_sorted_idx])
        if show_plot:
            indices = np.array(range(0, len(fi))) + 0.5
            fig, (ax1) = plt.subplots(1, 1, figsize=(12, 8))
            ax1.barh(indices, fi[importance_sorted_idx], height=0.7)
            ax1.set_yticklabels(feature_names[importance_sorted_idx])
            ax1.set_yticks(indices)
            ax1.set_ylim((0, len(fi)))
            plt.title(f'{model_name.upper()} Feature Importances')
            fig.tight_layout()
            plt.show()


def train_model(model_name='random_forest', verbose=False):
    X_train, X_test, y_train, y_test = get_data(num_days_per_month=7)

    # train model
    model = build_train_model(model_name, X_train, y_train, vb=verbose)

    # evaluate model
    evaluate(model, X_test, y_test, model_name=model_name, plot_probs=True)

    # save model
    os.chdir("../../../../../../home/sdas11/")
    joblib.dump(model, f'{model_name}.model')

    feat_imp(model, model_name=model_name)


def train_all(verbose=False):
    X_train, X_test, y_train, y_test = get_data(num_days_per_month=7)

    for model_name in  ['NN', 'random_forest', 'xgboost_clf', 'logistic_regression', 'svm', 'naive_bayes']:
        print(model_name.upper(), ':')
        # train model
        model = build_train_model(model_name, X_train, y_train, vb=verbose)

        # evaluate model
        evaluate(model, X_test, y_test, model_name=model_name, plot_probs=False)

        # save model
        os.chdir("../../../../../../home/sdas11/")
        joblib.dump(model, f'{model_name}.model')
        print('\n')
        feat_imp(model, model_name=model_name, show_plot=False)
        print('\n')

name = sys.argv[1]
train_all() if name == 'all' else train_model(model_name=name, verbose=True)

