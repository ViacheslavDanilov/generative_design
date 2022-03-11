import os
import logging
import argparse
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd

from tools.model import Regressor

os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%d.%m.%Y %I:%M:%S',
    filename=f'logs/{Path(__file__).stem}.log',
    filemode='w',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def main(
        data: Union[List[int], np.ndarray, pd.DataFrame, str],
        model_paths: Union[List[str], str],
        save_dir: str,
) -> None:

    model_paths = [model_paths, ] if isinstance(model_paths, str) else model_paths

    regressor_vms = Regressor(model_paths[0])            # FIXME: load several models

    if (
            isinstance(data, str)
            and Path(data).suffix == '.xlsx'
    ):
        data = pd.read_excel(data)
        data = data.values
    elif isinstance(data, np.ndarray):
        pass
    elif isinstance(data, pd.DataFrame):
        data = data.values
    elif isinstance(data, list):
        data = np.array(data, dtype=float)
        data = np.expand_dims(data, axis=0) if len(data.shape) == 1 else data
    else:
        raise ValueError('Unsupported data type')

    vms = regressor_vms(data)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Dataset conversion')
    parser.add_argument('--data', default='dataset/data_test.xlsx')
    parser.add_argument('--model_paths', nargs='+', type=str)
    parser.add_argument('--save_dir', default='experiments', type=str)
    args = parser.parse_args()

    main(
        data=args.data,
        model_paths=args.model_paths,
        save_dir=args.save_dir,
    )

    logger.info('')
    logger.info('Model prediction complete')
