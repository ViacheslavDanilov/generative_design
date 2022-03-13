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
logger = logging.getLogger(__name__)        # TODO: fix logger


def main(
    data: Union[List[int], np.ndarray, pd.DataFrame, str],
    lumen_model_path: str,
    vms_model_path: str,
    save_dir: str,
) -> None:

    if (
            isinstance(data, str)
            and Path(data).suffix == '.xlsx'
    ):
        data = pd.read_excel(data)
        features = list(data.columns)
        data = data.values
    elif isinstance(data, np.ndarray):
        features = list(map(lambda x: f'F{x}', range(data.shape[1])))
        pass
    elif isinstance(data, pd.DataFrame):
        data = data.values
        features = list(map(lambda x: f'F{x}', range(data.shape[1])))
    elif isinstance(data, list):
        data = np.array(data, dtype=float)
        features = list(map(lambda x: f'F{x}', range(data.shape[1])))
        data = np.expand_dims(data, axis=0) if len(data.shape) == 1 else data
    else:
        raise ValueError('Unsupported data type')

    # TODO: Add support of single model prediction
    logger.info('Point 0')
    model_lumen = Regressor(lumen_model_path)
    model_vms = Regressor(vms_model_path)
    logger.info('Point 1')
    lumen = model_lumen(data)
    logger.info('Point 2')
    vms = model_vms(data)
    logger.info('Point 3')
    data = np.hstack([data, lumen, vms])
    df_out = pd.DataFrame(data, columns=[*features, 'Lumen', 'VMS'])
    df_out.to_excel(
        os.path.join(save_dir, 'predictions.xlsx'),
        sheet_name='Predictions',
        index=True,
        index_label='Design',
        startrow=0,
        startcol=0,
    )

    logger.info('')
    logger.info('Prediction complete')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Dataset conversion')
    parser.add_argument('--data', default='dataset/data_test.xlsx')
    parser.add_argument('--lumen_model_path', type=str)
    parser.add_argument('--vms_model_path', type=str)
    parser.add_argument('--save_dir', default='calculations', type=str)
    args = parser.parse_args()

    main(
        data=args.data,
        lumen_model_path=args.lumen_model_path,
        vms_model_path=args.vms_model_path,
        save_dir=args.save_dir,
    )
