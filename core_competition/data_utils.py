import argparse

import h5py
import numpy as np
import torch
import torch.nn.functional as F  # noqa
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    def __init__(self, args: argparse.Namespace, files: list, static_input_dict: dict):
        self.args = args
        self.files = files
        self.static_input_dict = static_input_dict

    def __getitem__(self, index):
        file_index = index // 241
        start_index = index % 241

        file_name = self.files[file_index]
        city = file_name.split('_')[-2]

        with h5py.File(file_name, 'r') as file_in:
            data = file_in.get('array')
            data = np.array(data[start_index: start_index + 24])

        dynamic_input, target = data[:12], data[[12, 13, 14, 17, 20, 23]]
        static_input = self.static_input_dict[city]

        return dynamic_input, static_input, target

    def __len__(self):
        return len(self.files) * 241


class ValidDataset(Dataset):
    def __init__(self, args: argparse.Namespace, valid_data: np.ndarray):
        self.args = args
        self.valid_data = valid_data

    def __getitem__(self, index):
        dynamic_input = self.valid_data[index, :, :, :, :]
        return dynamic_input

    def __len__(self):
        return self.valid_data.shape[0]


def train_collate_fn(batch):
    dynamic_input_batch, static_input_batch, target_batch = zip(*batch)
    dynamic_input_batch = np.stack(dynamic_input_batch, axis=0)
    static_input_batch = np.stack(static_input_batch, axis=0)
    target_batch = np.stack(target_batch, axis=0)
    dynamic_input_batch = np.moveaxis(dynamic_input_batch, source=4, destination=2)
    dynamic_input_batch = torch.from_numpy(dynamic_input_batch).float()
    static_input_batch = torch.from_numpy(static_input_batch)
    target_batch = np.moveaxis(target_batch, source=4, destination=2)
    target_batch = torch.from_numpy(target_batch).float()
    return dynamic_input_batch, static_input_batch, target_batch


def valid_collate_fn(batch):
    dynamic_input_batch = batch
    dynamic_input_batch = np.stack(dynamic_input_batch, axis=0)
    dynamic_input_batch = np.moveaxis(dynamic_input_batch, source=4, destination=2)
    dynamic_input_batch = torch.from_numpy(dynamic_input_batch).float()
    return dynamic_input_batch
