import os
import time
import logging
import argparse
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd

from tools.utils import calculate_design_score

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


def predict(
    data: Union[List[float], np.ndarray, pd.DataFrame, str],
    lumen_model_path: str,
    stress_model_path: str,
    features: List[str],
    uts: float,
    save_dir: str,
) -> None:

    if (
            isinstance(data, str)
            and Path(data).suffix == '.xlsx'
    ):
        data = pd.read_excel(data)
        data = data[features].values
    elif isinstance(data, np.ndarray):
        pass
    elif isinstance(data, pd.DataFrame):
        data = data.values
    elif isinstance(data, list):
        data = np.array(data, dtype=float)
    else:
        raise ValueError('Unsupported data type')

    data = np.expand_dims(data, axis=0) if len(data.shape) == 1 else data

    logger.info('')
    logger.info(f'Model (Lumen).............: {lumen_model_path}')
    logger.info(f'Model (Stress)............: {stress_model_path}')

    lumen = np.empty((data.shape[0], 1))
    lumen[:] = np.NaN
    if (
            isinstance(lumen_model_path, str)
            and lumen_model_path is not None
    ):
        start = time.time()
        model_lumen = Regressor(
            model_path=lumen_model_path,
        )
        lumen = model_lumen(data)
        end = time.time()
        logger.info(f'Lumen prediction took.....: {end - start:.0f} seconds')

    stress = np.empty((data.shape[0], 1))
    stress[:] = np.NaN
    if (
            isinstance(stress_model_path, str)
            and stress_model_path is not None
    ):
        start = time.time()
        model_stress = Regressor(
            model_path=stress_model_path,
        )
        stress = model_stress(data)
        end = time.time()
        logger.info(f'Stress prediction took....: {end - start:.0f} seconds')

    score = np.empty((data.shape[0], 1))
    score[:] = np.NaN
    for idx, (_lumen, _stress) in enumerate(zip(lumen, stress)):
        _lumen = float(_lumen.squeeze())
        _stress = float(_stress.squeeze())
        score[idx] = calculate_design_score(
            lumen_abs=_lumen,
            stress_abs=_stress,
            uts=uts,
        )

    data = np.hstack([data, lumen, stress, score])
    df_out = pd.DataFrame(data, columns=[*features, 'Lumen', 'Stress', 'Score'])
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'predictions.xlsx')
    df_out.index += 1
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

    FEATURES = [
        'HGT',
        'DIA',
        'ANG',
        'CVT',
        'THK',
        'ELM',
    ]

    parser = argparse.ArgumentParser(description='Model inference')
    parser.add_argument('--data', default='dataset/test.xlsx', nargs='+')
    parser.add_argument('--lumen_model_path', default=None, type=str)
    parser.add_argument('--stress_model_path', default=None, type=str)
    parser.add_argument('--features', default=FEATURES, nargs='+', type=str)
    parser.add_argument('--uts', default=8.9, type=float)
    parser.add_argument('--save_dir', default='experiments/predict', type=str)
    args = parser.parse_args()

    predict(
        data=args.data,
        lumen_model_path=args.lumen_model_path,
        stress_model_path=args.stress_model_path,
        features=args.features,
        uts=args.uts,
        save_dir=args.save_dir,
    )
