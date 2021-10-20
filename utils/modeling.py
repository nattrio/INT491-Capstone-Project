from mlxtend.plotting import plot_confusion_matrix
from utils.preprocess import *

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import scikitplot as skplt


def acceptance(matrix, n):
    i = n-1
    if n == 1:
        score = matrix[i][i] + matrix[i+1][i]
    elif n == len(matrix):
        score = matrix[i-1][i] + matrix[i][i]
    else:
        score = matrix[i-1][i] + matrix[i][i] + matrix[i+1][i]
    return score


def nan_to_zero(n):
    if np.isnan(n):
        return 0
    else:
        return n


def eval_predict(array):
    temp = np.asmatrix(array).transpose()
    matrix = np.array(temp)
    for i in range(len(matrix)):
        class_num = i+1
        print("Predict {} ({})".format(class_num, matrix[i].sum()))
        for n in range(len(matrix[i])):
            pred_num = n+1
            score = matrix[i][n]
            # print(" - Actual Class {}: {}".format(pred_num, score))
            if i == n:
                print(" - Accuracy:\t{}\t({:.0%})".format(score,
                      nan_to_zero(score/matrix[i].sum())))
        acp = acceptance(array, class_num)
        print(" - Acceptable:\t{}\t({:.0%})".format(nan_to_zero(acp),
              nan_to_zero(acp/matrix[i].sum())))
        print(" - Missed:\t{}\t({:.0%})".format(
            matrix[i].sum() - acp, 1 - nan_to_zero(acp/matrix[i].sum())))


def kfold_eval(n_splits, model, X, y_class):
    model_score = []
    underEst_score = []
    train_score = []
    
    train_conf = []
    validate_conf = []
    
    pred_all = np.array(())
    cv = KFold(n_splits)

    for train_index, test_index in cv.split(X):
        X_train_k, y_train_k = X[train_index], y_class[train_index]
        X_test_k, y_test_k = X[test_index], y_class[test_index]

        # print("test index: {}-{}".format(test_index[0], test_index[-1]))

        model.fit(X_train_k, y_train_k)
        
        train_pred = model.predict(X_train_k)
        train_accur = accuracy_score(y_train_k, train_pred)
        
        pred = model.predict(X_test_k)
        accur = accuracy_score(y_test_k, pred)
        underEst = underEstimate_cal(y_test_k, pred)

        model_score.append(accur)
        underEst_score.append(underEst)
        train_score.append(train_accur)
        
              
        pred_all = np.concatenate((pred_all, pred), axis=None)
        # skplt.metrics.plot_confusion_matrix(y_test_k, pred)
        
        train_conf.append(confusion_matrix(y_train_k, train_pred))
        validate_conf.append(confusion_matrix(y_test_k, pred))
        
    howFitting(train_score, model_score)
    
    show_score("Train score", train_score)
    showAllConfusion(train_conf)
    
    show_score("Validation", model_score)
    showAllConfusion(validate_conf)
    plot_confusion_matrix(conf_mat=confusion_matrix(y_class, pred_all))
    
    show_score("Under estimate", underEst_score)
    pred_all = pred_all.astype(int)
    eval_predict(confusion_matrix(y_class, pred_all))
    
    


def show_score(title, score):
    print(title)
    print("Mean:\t", np.mean(score))
    # print("Max:\t", np.max(score))
    # print("Min:\t", np.min(score))
    # print("S.D.:\t", np.std(score))
    print(score)

# Linear model


def kfold_eval_LR(n_splits, model, model_type, X, y, y_class, df, OUT_CLASS, OUT_SCALE):
    lr_score = []
    pred_all = np.array(())
    cv = KFold(n_splits)
    scale_sit = scaleList(df["SIT GPAX in 1/59"], OUT_CLASS, OUT_SCALE)

    for train_index, test_index in cv.split(X):
        if model_type == "LR-normal":
            X_train_k, y_train_k = X[train_index], y_class[train_index]
            X_test_k, y_test_k = X[test_index], y_class[test_index]
            y_train_c, y_test_c = y[train_index], y[test_index]
            model.fit(X_train_k, y_train_c)

        elif model_type == "LR-100":
            X_train_k, y_train_k = X[train_index], y_class[train_index]
            X_test_k, y_test_k = X[test_index], y_class[test_index]
            model.fit(X, y)

        pred = model.predict(X_test_k)
        for i in range(len(pred)):
            pred[i] = funcClass(pred[i], scale_sit)

        accur = accuracy_score(y_test_k, pred)
        lr_score.append(accur)
        pred_all = np.concatenate((pred_all, pred), axis=None)
    show_score("Validation", lr_score)
    eval_predict(confusion_matrix(y_class, pred_all))
    skplt.metrics.plot_confusion_matrix(y_class, pred_all)

# Return only model score


def underEstimate_cal(y_true, y_pred):
    count_score = 0
    for i in range(len(y_true)):
        if y_pred[i] == y_true[i] or y_pred[i]+1 == y_true[i]:
            count_score += 1
    return count_score/len(y_true)
    

def kfold_score(n_splits, model, X, y_class):
    model_score = []
    underEst_score = []
    
    pred_all = np.array(())
    cv = KFold(n_splits)

    for train_index, test_index in cv.split(X):
        X_train_k, y_train_k = X[train_index], y_class[train_index]
        X_test_k, y_test_k = X[test_index], y_class[test_index]

        model.fit(X_train_k, y_train_k)
        pred = model.predict(X_test_k)
        accur = accuracy_score(y_test_k, pred)
        underEst = underEstimate_cal(y_test_k, pred)

        model_score.append(accur)
        underEst_score.append(underEst)
        pred_all = np.concatenate((pred_all, pred), axis=None)
    return np.mean(model_score), np.mean(underEst_score)
