import argparse
import os
import random

import h5py
import numpy as np
import torch
import torch.nn.functional as F  # noqa
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import TRAINING_CITIES, CORE_CHALLENGE_CITIES
from data_utils import TrainDataset, train_collate_fn
from model import UNet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--in_channels', type=int, default=12 * 8)
    parser.add_argument('--out_channels', type=int, default=6 * 8)
    parser.add_argument('--hidden_channels', type=list, default=[128, 256])
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--padding', type=int, default=1)
    parser.add_argument('--num_groups', type=int, default=8)
    parser.add_argument('--use_static_input', type=bool, default=True)
    parser.add_argument('--input_normalization', type=bool, default=True)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'RMSProp'], default='Adam')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--device', type=int, default=None)
    return parser.parse_args()


def train(args: argparse.Namespace):
    static_input_dict = dict()
    for city in TRAINING_CITIES + CORE_CHALLENGE_CITIES:
        with h5py.File(f'../data/{city}/{city}_static.h5', 'r') as file_in:
            static_input = file_in.get('array')
            static_input = np.array(static_input, dtype=np.float32)
            static_input_dict[city] = static_input
    if args.input_normalization:
        static_input = np.stack(list(static_input_dict.values()), axis=0)
        static_input_mean = static_input.mean(axis=(0, 2, 3))[:, None, None]
        static_input_std = static_input.std(axis=(0, 2, 3))[:, None, None]
        for city in static_input_dict:
            static_input_dict[city] = (static_input_dict[city] - static_input_mean) / static_input_std
    if args.input_normalization:
        dynamic_input_mean = np.load('../cache/dynamic_input_mean.npy')
        dynamic_input_std = np.load('../cache/dynamic_input_std.npy')
        dynamic_input_mean = torch.from_numpy(dynamic_input_mean)[None, None, :, None, None].float().cuda()
        dynamic_input_std = torch.from_numpy(dynamic_input_std)[None, None, :, None, None].float().cuda()
    else:
        dynamic_input_mean = dynamic_input_std = None

    train_files = []
    for city in TRAINING_CITIES + CORE_CHALLENGE_CITIES:
        data_files = [file for file in f'../data/{city}/training' if file.endswith('_8ch.h5')]
        data_files = [f'../data/{city}/training/{file}' for file in data_files]
        train_files.extend(data_files)
    train_dataset = TrainDataset(args=args, files=train_files, static_input_dict=static_input_dict)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=train_collate_fn,
        pin_memory=True,
        drop_last=True
    )

    model = UNet(args=args)

    if args.optimizer == 'SGD':
        optimizer = optim.SGD(
            params=model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'Adam':
        optimizer = optim.Adam(
            params=model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'RMSProp':
        optimizer = optim.RMSprop(
            params=model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            momentum=args.momentum
        )
    else:
        raise ValueError()

    model.cuda()

    global_step = 0

    for epoch in range(1, args.num_epochs + 1):
        for dynamic_input_batch, static_input_batch, target_batch in tqdm(train_loader):
            dynamic_input_batch = dynamic_input_batch.cuda()
            static_input_batch = static_input_batch.cuda()
            target_batch = target_batch.cuda()
            if args.input_normalization:
                dynamic_input_batch = (dynamic_input_batch - dynamic_input_mean) / dynamic_input_std
            dynamic_input_batch = dynamic_input_batch.reshape(-1, 96, 495, 436)
            dynamic_input_batch = F.pad(dynamic_input_batch, pad=(6, 6, 1, 0))
            target_batch = target_batch.reshape(-1, 48, 495, 436)
            if args.use_static_input:
                static_input_batch = F.pad(static_input_batch, pad=(6, 6, 1, 0))
                input_batch = torch.cat([dynamic_input_batch, static_input_batch], dim=1)
            else:
                input_batch = dynamic_input_batch
            prediction_batch = model(x=input_batch)
            optimizer.zero_grad()
            loss = F.mse_loss(input=prediction_batch[:, :, 1:, 6: -6], target=target_batch)
            loss.backward()
            optimizer.step()
            global_step += 1
            if global_step == 50_000:
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }, os.path.join(f'../model/{args.model_name}.bin'))
                return


def main():
    torch.backends.cudnn.benchmarks = True  # noqa

    args = parse_args()

    assert args.model_name is not None

    if args.use_static_input:
        args.in_channels += 9

    if args.device is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    train(args=args)


if __name__ == '__main__':
    main()
