import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from tools.metrics import calculate_all_metrics

os.makedirs('../logs', exist_ok=True)
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%d.%m.%Y %I:%M:%S',
    filename='logs/{:s}.log'.format(Path(__file__).stem),
    filemode='w',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def get_optimizer_name(
    path: str,
) -> str:
    if 'CMA' in path:
        optimizer = 'CMA'
    elif 'MOTPE' in path:
        optimizer = 'CMA'
    elif 'NSGA' in path:
        optimizer = 'CMA'
    elif 'QMC' in path:
        optimizer = 'CMA'
    elif 'RS' in path:
        optimizer = 'CMA'
    elif 'TPE' in path:
        optimizer = 'CMA'
    else:
        optimizer = 'UNK'

    return optimizer


def main(
    data_path: str,
    save_dir: str,
) -> None:
    df = pd.read_excel(data_path)
    optimizer = get_optimizer_name(data_path)

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{optimizer}_metrics.xlsx')
    writer = pd.ExcelWriter(save_path, engine='xlsxwriter')

    for target in ['LMN', 'STS']:
        df_target = df[[f'{target}_GT', f'{target}_Pred']]
        y_true = np.array(df_target[f'{target}_GT'])
        y_pred = np.array(df_target[f'{target}_Pred'])
        metrics = calculate_all_metrics(y_true, y_pred)
        df_metrics = pd.DataFrame(metrics.items())
        df_metrics.index += 1
        df_metrics.to_excel(
            writer,
            sheet_name=f'{target}',
            index=True,
            index_label='ID',
        )
    writer.save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset conversion')
    parser.add_argument('--data_path', default='experiments/tune/CMA_abaqus_done.xlsx', type=str)
    parser.add_argument('--save_dir', default='experiments/tune', type=str)
    args = parser.parse_args()

    main(
        data_path=args.data_path,
        save_dir=args.save_dir,
    )
