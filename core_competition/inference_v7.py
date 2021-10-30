import argparse
import gc
import os
import zipfile

import h5py
import numpy as np
import torch
import torch.nn.functional as F  # noqa
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import TRAINING_CITIES, CORE_CHALLENGE_CITIES
from data_utils import ValidDataset, valid_collate_fn
from file_utils import write_data_to_h5
from model_v7 import UNet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--submission_name', type=str, default=None)
    parser.add_argument('--in_channels', type=int, default=12 * 8)
    parser.add_argument('--out_channels', type=int, default=6 * 8)
    parser.add_argument('--hidden_channels', type=list, default=[128, 256, 512, 1024, 2048])
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--padding', type=int, default=1)
    parser.add_argument('--num_groups', type=int, default=8)
    parser.add_argument('--use_static_input', type=bool, default=True)
    parser.add_argument('--input_normalization', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--device', type=int, default=None)
    return parser.parse_args()


def inference(args: argparse.Namespace):
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
            static_input = torch.from_numpy((static_input_dict[city] - static_input_mean) / static_input_std).float()
            static_input = F.pad(static_input, pad=(6, 6, 1, 0))
            static_input = static_input.cuda()
            static_input_dict[city] = static_input
    if args.input_normalization:
        dynamic_input_mean = np.load('../cache/dynamic_input_mean.npy')
        dynamic_input_std = np.load('../cache/dynamic_input_std.npy')
        dynamic_input_mean = torch.from_numpy(dynamic_input_mean)[None, None, :, None, None].float().cuda()
        dynamic_input_std = torch.from_numpy(dynamic_input_std)[None, None, :, None, None].float().cuda()
    else:
        dynamic_input_mean = dynamic_input_std = None

    model = UNet(args=args)
    model.load_state_dict(torch.load(args.checkpoint_path)['model'])
    model.cuda()
    model.eval()

    with zipfile.ZipFile(f'../submission/{args.submission_name}.zip', 'w') as z:
        for city in CORE_CHALLENGE_CITIES:
            with h5py.File(f'../data/{city}/{city}_test_temporal.h5', 'r') as file_in:
                valid_data = file_in.get('array')
                valid_data = np.array(valid_data)
            valid_dataset = ValidDataset(args=args, valid_data=valid_data)
            valid_loader = DataLoader(
                dataset=valid_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                collate_fn=valid_collate_fn,
                pin_memory=True,
                drop_last=False
            )
            valid_predictions = []
            with torch.no_grad():
                for dynamic_input_batch in tqdm(valid_loader):
                    batch_size = dynamic_input_batch.size(0)
                    dynamic_input_batch = dynamic_input_batch.cuda()
                    if args.input_normalization:
                        dynamic_input_batch = (dynamic_input_batch - dynamic_input_mean) / dynamic_input_std
                    dynamic_input_batch = dynamic_input_batch.reshape(-1, 96, 495, 436)
                    dynamic_input_batch = F.pad(dynamic_input_batch, pad=(6, 6, 1, 0))
                    if args.use_static_input:
                        static_input_batch = static_input_dict[city].repeat(batch_size, 1, 1, 1)
                        input_batch = torch.cat([dynamic_input_batch, static_input_batch], dim=1)
                    else:
                        input_batch = dynamic_input_batch
                    prediction_batch = model(x=input_batch)
                    valid_predictions.append(prediction_batch.cpu().numpy())
            valid_predictions = np.concatenate(valid_predictions, axis=0).clip(min=0.0, max=255.0).astype(np.float32)
            valid_predictions = valid_predictions[:, :, 1:, 6: -6]
            valid_predictions = valid_predictions.reshape(-1, 6, 8, 495, 436)
            valid_predictions = np.moveaxis(valid_predictions, source=2, destination=4)
            np.save(f'../submission/{args.submission_name}_{city}_predictions.npy', valid_predictions)
            write_data_to_h5(
                data=valid_predictions,
                filename=f'../submission/{args.submission_name}_{city}_predictions.h5',
                compression_level=6
            )
            z.write(
                f'../submission/{args.submission_name}_{city}_predictions.h5',
                arcname=f'{city}/{city}_test_temporal.h5'
            )


def main():
    args = parse_args()

    if args.use_static_input:
        args.in_channels += 9

    if args.device is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)

    assert args.submission_name is not None and args.checkpoint_path is not None

    inference(args=args)


if __name__ == '__main__':
    main()
