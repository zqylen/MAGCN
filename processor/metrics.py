# -*- coding:utf-8 -*-
import csv
import numpy as np

def get_adjacency_matrix(distance_df_filename, num_of_vertices):
    '''
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information

    num_of_vertices: int, the number of vertices

    Returns
    ----------
    A: np.ndarray, adjacency matrix

    '''

    with open(distance_df_filename, 'r') as f:
        reader = csv.reader(f)
        header = f.__next__()
        edges = [(int(i[0]), int(i[1])) for i in reader]

    A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                 dtype=np.float32)

    for i, j in edges:
        A[i, j] = 1

    return np.expand_dims(A, 0)

def masked_mape_np(y_true, y_pred, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide(np.subtract(y_pred, y_true).astype('float32'),
                      y_true))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100


def mean_absolute_error(y_true, y_pred):
    '''
    mean absolute error

    Parameters
    ----------
    y_true, y_pred: np.ndarray, shape is (batch_size, num_of_features)

    Returns
    ----------
    np.float64

    '''

    return np.mean(np.abs(y_true - y_pred))


def mean_squared_error(y_true, y_pred):
    '''
    mean squared error

    Parameters
    ----------
    y_true, y_pred: np.ndarray, shape is (batch_size, num_of_features)

    Returns
    ----------
    np.float64

    '''
    return np.mean((y_true - y_pred) ** 2)
