import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from utils.scaler import *
from utils.visualize import *

def dropRecords(df, feature):
    print("Before:\t", df['Yr'].unique())
    print(df.shape)

    # Drop null subject
    df.dropna(subset=feature, inplace=True)

    # Drop Yr <= 51
    # df.drop(df[df['Yr'] <= 51].index, axis=0, inplace=True)

    # หลังจาก drop แล้ว ให้เรียง index ใหม่
    df.index = range(len(df))
    print("After:\t", df['Yr'].unique())
    print(df.shape)
    return df

def toClassMethod(col_list, method, n_class, df):
        if n_class <= 1:
            for col in col_list:
                visualizeScale(df[col], col, [0])
        else:
            for col in col_list:
                scale_temp = scaleList(df[col], n_class, method)
                print(col, scale_temp)
                applyClass(df[col], col, scale_temp)
                
def toClassCustom(col_list, scale, df):
            for col in col_list:
                print(col, scale)
                applyClass(df[col], col, scale)


def scaleList(df_list, n_class, scale):
    sorted_df_list = sorted(df_list)
    sample = len(sorted_df_list)
    score_range = sorted_df_list[-1]-sorted_df_list[0]

    if scale == 'score_range':
        scale_class = scoreRangeScale(sorted_df_list, score_range, n_class)
        return scale_class

    elif scale == 'sample_size':
        scale_class = sampleSizeScale(sorted_df_list, sample, n_class)
        return scale_class


def applyClass(df_col, col, scale_list):
    visualizeScale(df_col, col, scale_list)

    for i in range(len(df_col)):
        df_col[i] = funcClass(df_col[i], scale_list)
    df_col = pd.to_numeric(df_col)


def funcClass(score, scale_list):
    for i in range(len(scale_list)):
        if i == len(scale_list)-1:
            return len(scale_list)
        elif score < scale_list[i]:
            return i+1


def assignXY(df, VAR_NUM):
    X = np.array(df[VAR_NUM])  # .reshape(1,-1)
    y = np.array(df['SIT GPAX in 1/59'])
    y_class = np.array(df['SIT_class'])
    return X, y, y_class


def splitData(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8)  # train size 0.8 คือ แบ่งไว้เทรน 80%
    print("All:", len(X))
    print("Train: {}, Test: {}".format(len(X_train), len(X_test)))
    return X_train, X_test, y_train, y_test
