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


def main(
    data_path: str,
    save_dir: str,
    optimizer: str = 'TPE',
    alpha: str = None,
) -> None:
    df = pd.read_excel(data_path)

    if optimizer is not None:
        df = df[df['Optimizer'] == optimizer]

    if alpha is not None:
        save_path = os.path.join(save_dir, f'{optimizer}_{alpha:.1f}_metrics.xlsx')
        df = df[df['Alpha'] == alpha]
    else:
        save_path = os.path.join(save_dir, f'{optimizer}_metrics.xlsx')

    os.makedirs(save_dir, exist_ok=True)
    writer = pd.ExcelWriter(save_path, engine='xlsxwriter')

    for target in ['LMN', 'STS']:
        y_true = np.array(df[f'{target}_true'])
        y_pred = np.array(df[f'{target}_pred'])
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
    parser.add_argument('--data_path', default='experiments/verification/Verification_results.xlsx', type=str)
    parser.add_argument('--optimizer', default='TPE', type=str, choices=['CMA', 'MOTPE', 'NSGA', 'QMC', 'RS', 'TPE'])
    parser.add_argument('--alpha', default=None, type=float, choices=[0.2, 0.5, 1.0, 2.0, 5.0])
    parser.add_argument('--save_dir', default='experiments/verification', type=str)
    args = parser.parse_args()

    main(
        data_path=args.data_path,
        optimizer=args.optimizer,
        alpha=args.alpha,
        save_dir=args.save_dir,
    )
