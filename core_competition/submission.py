import zipfile

import numpy as np
from tqdm import trange

from config import CORE_CHALLENGE_CITIES
from file_utils import write_data_to_h5


def main():
    with zipfile.ZipFile('../submission/core_submission.zip', 'w') as z:
        for city in CORE_CHALLENGE_CITIES:
            city_predictions = None
            for submission_index in trange(1, 10):
                if city_predictions is None:
                    city_predictions = np.load(f'../submission/model_v{submission_index}_{city}_predictions.npy')
                else:
                    city_predictions += np.load(f'../submission/model_v{submission_index}_{city}_predictions.npy')
            city_predictions /= 9.0
            city_predictions = np.clip(city_predictions, a_min=0.0, a_max=255.0)
            write_data_to_h5(
                data=city_predictions,
                filename=f'../submission/core_submission_{city}_predictions.h5',
                compression_level=6
            )
            z.write(
                f'../submission/core_submission_{city}_predictions.h5',
                arcname=f'{city}/{city}_test_temporal.h5'
            )


if __name__ == '__main__':
    main()
