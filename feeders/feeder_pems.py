import os
import sys
import numpy as np
import random
import pickle
import json
import torch
import torch.nn as nn
from torchvision import datasets, transforms


def search_data(sequence_length, num_of_depend, label_start_idx,
                num_for_predict, units, points_per_hour):
    '''
    Parameters
    ----------
    sequence_length: int, length of all history data
    num_of_depend: int,
    label_start_idx: int, the first index of predicting target
    num_for_predict: int, the number of points will be predicted for each sample
    units: int, week: 7 * 24, day: 24, recent(hour): 1
    points_per_hour: int, number of points per hour, depends on data
    Returns
    ----------
    list[(start_idx, end_idx)]
    '''

    if points_per_hour < 0:
        raise ValueError("points_per_hour should be greater than 0!")

    if label_start_idx + num_for_predict > sequence_length:
        return None

    x_idx = []
    for i in range(1, num_of_depend + 1):
        start_idx = label_start_idx - points_per_hour * units * i
        end_idx = start_idx + num_for_predict
        if start_idx >= 0:
            x_idx.append((start_idx, end_idx))
        else:
            return None

    if len(x_idx) != num_of_depend:
        return None

    return x_idx[::-1]


def get_sample_indices(data_sequence, num_of_weeks, num_of_days, num_of_hours,
                       label_start_idx, num_for_predict, points_per_hour=12):
    '''
    Parameters
    ----------
    data_sequence: np.ndarray
                   shape is (sequence_length, num_of_vertices, num_of_features)

    num_of_weeks, num_of_days, num_of_hours: int

    label_start_idx: int, the first index of predicting target

    num_for_predict: int,
                     the number of points will be predicted for each sample

    points_per_hour: int, default 12, number of points per hour

    Returns
    ----------
    week_sample: np.ndarray
                 shape is (num_of_weeks * points_per_hour,
                           num_of_vertices, num_of_features)

    day_sample: np.ndarray
                 shape is (num_of_days * points_per_hour,
                           num_of_vertices, num_of_features)

    hour_sample: np.ndarray
                 shape is (num_of_hours * points_per_hour,
                           num_of_vertices, num_of_features)

    target: np.ndarray
            shape is (num_for_predict, num_of_vertices, num_of_features)
    '''
    week_sample, day_sample, hour_sample = None, None, None

    week_indices = search_data(data_sequence.shape[0], num_of_weeks,
                               label_start_idx, num_for_predict,
                               7 * 24, points_per_hour)

    day_indices = search_data(data_sequence.shape[0], num_of_days,
                              label_start_idx, num_for_predict,
                              24, points_per_hour)

    hour_indices = search_data(data_sequence.shape[0], num_of_hours,
                               label_start_idx, num_for_predict,
                               1, points_per_hour)
    if hour_indices:
        hour_sample = np.concatenate([data_sequence[i: j]
                                      for i, j in hour_indices], axis=0)
    if day_indices:
        day_sample = np.concatenate([data_sequence[i: j]
                                     for i, j in day_indices], axis=0)
    # elif num_of_days and hour_indices:
    #     day_sample = np.concatenate([data_sequence[i: j]
    #                                  for i, j in hour_indices], axis=0)
    if week_indices:
        week_sample = np.concatenate([data_sequence[i: j]
                                      for i, j in week_indices], axis=0)
    # elif num_of_weeks and hour_indices:
    #     week_sample = np.concatenate([data_sequence[i: j]
    #                                   for i, j in hour_indices], axis=0)

    target = data_sequence[label_start_idx: label_start_idx + num_for_predict]

    return hour_sample, day_sample, week_sample, target


class Feeder_pems(torch.utils.data.Dataset):
    """ Feeder for skeleton-based action recognition in kinetics-skeleton dataset
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
        label_path: the path to label
        random_choose: If true, randomly choose a portion of the input sequence
        random_shift: If true, randomly pad zeros at the begining or end of sequence
        random_move: If true, perform randomly but continuously changed transformation to input sequence
        window_size: The length of the output sequence
        pose_matching: If ture, match the pose between two frames
        num_person_in: The number of people the feeder can observe in the input sequence
        num_person_out: The number of people the feeder in the output sequence
        debug: If true, only use the first 100 samples
    """

    def __init__(self,
                 data_path,
                 num_of_weeks=0,
                 num_of_days=0,
                 num_of_hours=1,
                 num_for_predict=12,
                 points_per_hour=12,
                 debug=False,
                 ):
        self.debug = debug
        self.data_path = data_path

        self.num_of_weeks = num_of_weeks
        self.num_of_days = num_of_days
        self.num_of_hours = num_of_hours
        self.num_for_predict = num_for_predict
        self.points_per_hour = points_per_hour
        self.load_data()

    def load_data(self):

        self.data = np.load(self.data_path)['data']  # (sequence_length, num_of_vertices, num_of_features)
        # output data shape (N, C, T, V, M)
        self.N = len(self.data)  #sample
        self.C = 3  #channel num_of_features
        self.T = self.num_for_predict  #time period
        self.V = 307  #vertices
        self.M = bool(self.num_of_weeks) + bool(self.num_of_days) + bool(self.num_of_hours) #used group

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return self

    def __getitem__(self, index):

        sample = get_sample_indices(self.data, self.num_of_weeks, self.num_of_days,
                                    self.num_of_hours, index, self.num_for_predict,
                                    self.points_per_hour)

        if sample[0] is None or sample[1] is None and self.num_of_days or sample[2] is None and self.num_of_weeks:
            return None, None

        hour_sample, day_sample, week_sample, label = sample

        sample = []  # [(hour_sample),(day_sample),(week_sample),target]
        if self.num_of_hours > 0:
            sample.append(hour_sample)#(T, V, C)
        if self.num_of_days > 0:
                sample.append(day_sample)
        if self.num_of_weeks > 0:
            sample.append(week_sample)



        label = label[:, :, 0]  # (T, V)

        return sample, label  # sampe：[(hour_sample),(day_sample),(week_sample)] = [(Th, V, C),(Td, V, C),(Tw, V, C)]
                                # label: (1,V,Tpre),
