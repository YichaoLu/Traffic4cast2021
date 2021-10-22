import h5py
import numpy as np


def write_data_to_h5(data: np.ndarray, filename: str, compression='gzip', compression_level=9, dtype='uint8'):
    with h5py.File(filename, 'w', libver='latest') as f:
        f.create_dataset(
            'array',
            shape=data.shape,
            data=data,
            chunks=(1, *data.shape[1:]),
            dtype=dtype,
            compression=compression,
            compression_opts=compression_level
        )
