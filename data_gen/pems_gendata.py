
import os
import sys
import pickle
import argparse
import yaml
from tqdm import tqdm
import numpy as np
from numpy.lib.format import open_memmap

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from feeders.feeder_pems import Feeder_pems

arg_norm = {}
def normalization(data, arg_norm):
    '''
    Parameters
    ----------
    train, val, test: np.ndarray (B,M,T,V,C)
    Returns
    ----------
    stats: dict, two keys: mean and std
    train_norm, val_norm, test_norm: np.ndarray,
                                     shape is the same as original
    '''

    if arg_norm == {}:
        arg_norm['mean'] = data.mean(axis=(0, 1, 2, 3), keepdims=True)
        arg_norm['std'] = data.std(axis=(0, 1, 2, 3), keepdims=True)

    def normalize(x):
        return (x - arg_norm['mean']) / arg_norm['std']

    data_norm = normalize(data)

    return data_norm, arg_norm

def gendata(
        data_path,
        data_out_path,
        label_out_path,
        index_range,
        num_of_weeks=0,
        num_of_days=0,
        num_of_hours=1,
        num_for_predict=12,
        points_per_hour=12,
        ):

    feeder = Feeder_pems(
        data_path=data_path,
        num_of_weeks=num_of_weeks,
        num_of_days=num_of_days,
        num_of_hours=num_of_hours,
        num_for_predict=num_for_predict,
        points_per_hour=points_per_hour,
        )
    feeder_len = len(feeder)
    sample_index = range(feeder_len)[int(index_range[0] * feeder_len): int(index_range[1] * feeder_len)]

    sample_label = []
    sample_all = []
    global arg_norm
    for i in tqdm(sample_index):
        data, label = feeder[i]
        if not data:
            continue
        sample_all.append(np.array(data))
        sample_label.append(np.array(label))
    sample_all = np.array(sample_all)[:, :, :, :, 0:1]
    #sample_all = np.array(sample_all)
    print(sample_all.shape)
    sample_all, arg_norm = normalization(sample_all, arg_norm)

    fp = open_memmap(
        data_out_path,
        dtype='float32',
        mode='w+',
        shape=sample_all.shape)
    fp[:, :, :, :] = sample_all

    with open(label_out_path, 'wb') as f:
        pickle.dump((sample_index, list(sample_label)), f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Kinetics-skeleton Data Converter.')
    parser.add_argument(
        '--data_path', default='../data/PEMS03')
    parser.add_argument(
        '--out_folder', default='../data/PEMS03')
    parser.add_argument(
        '--config', default='../config/pems03/pems03_train.yaml')
    arg = parser.parse_args()

    #load config file
    with open(arg.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        data_config = config['gen_config_args']

    part = ['train', 'val']
    index_range = [[0, 0.6], [0.8, 1]]
    for p, ir in zip(part, index_range):
        print('Processing dataset_{}...'.format(p))
        data_path = '{}/{}.npz'.format(arg.data_path, arg.data_path.split('/')[-1])

        data_out_path = '{}/{}_data.npy'.format(arg.out_folder, p)
        label_out_path = '{}/{}_label.pkl'.format(arg.out_folder, p)

        if not os.path.exists(arg.out_folder):
            os.makedirs(arg.out_folder)

        gendata(data_path, data_out_path, label_out_path, index_range=ir,
                num_of_weeks=data_config['num_of_weeks'],
                num_of_days=data_config['num_of_days'],
                num_of_hours=data_config['num_of_hours'],
                num_for_predict=data_config['num_for_predict'],
                points_per_hour=data_config['points_per_hour'],)
