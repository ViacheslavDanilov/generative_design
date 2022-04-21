import os
import time
import logging
import argparse
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd

from tools.utils import score_func

os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%d.%m.%Y %I:%M:%S',
    filename=f'logs/{Path(__file__).stem}.log',
    filemode='w',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
from tools.model import Regressor


def main(
    data: Union[List[int], np.ndarray, pd.DataFrame, str],
    lumen_model_path: str,
    vms_model_path: str,
    use_golden_features: bool,
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

    logger.info('')
    logger.info(f'Model (Lumen).............: {lumen_model_path}')
    logger.info(f'Model (VMS)...............: {vms_model_path}')
    logger.info(f'Use golden features.......: {use_golden_features}')

    lumen = np.empty((data.shape[0], 1))
    lumen[:] = np.NaN
    if (
            isinstance(lumen_model_path, str)
            and lumen_model_path is not None
    ):
        start = time.time()
        model_lumen = Regressor(
            model_path=lumen_model_path,
            use_golden_features=use_golden_features,
        )
        lumen = model_lumen(data)
        end = time.time()
        logger.info(f'Lumen prediction took.....: {end - start:.0f} seconds')

    vms = np.empty((data.shape[0], 1))
    vms[:] = np.NaN
    if (
            isinstance(vms_model_path, str)
            and vms_model_path is not None
    ):
        start = time.time()
        model_vms = Regressor(
            model_path=vms_model_path,
            use_golden_features=use_golden_features,
        )
        vms = model_vms(data)
        end = time.time()
        logger.info(f'VMS prediction took.......: {end - start:.0f} seconds')

    material_UTS = 10
    score = score_func(lumen, vms, material_UTS)

    data = np.hstack([data, lumen, vms, score])
    df_out = pd.DataFrame(data, columns=[*features, 'Lumen', 'VMS', 'Score'])
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'predictions.xlsx')
    df_out.to_excel(
        save_path,
        sheet_name='Predictions',
        index=True,
        index_label='Design',
        startrow=0,
        startcol=0,
    )
    logger.info(f'Predictions saved to......: {save_path}')
    logger.info('Complete')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Dataset conversion')
    parser.add_argument('--data', default='dataset/data_test.xlsx')
    parser.add_argument('--lumen_model_path', default=None, type=str)
    parser.add_argument('--vms_model_path', default=None, type=str)
    parser.add_argument('--use_golden_features', action='store_true')
    parser.add_argument('--save_dir', default='calculations', type=str)
    args = parser.parse_args()

    main(
        data=args.data,
        lumen_model_path=args.lumen_model_path,
        vms_model_path=args.vms_model_path,
        use_golden_features=args.use_golden_features,
        save_dir=args.save_dir,
    )
