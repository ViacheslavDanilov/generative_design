import argparse
import logging
import os
import pickle
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import shap

os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%d.%m.%Y %I:%M:%S',
    filename='logs/{:s}.log'.format(Path(__file__).stem),
    filemode='w',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
from tools.model import Regressor


def main(
    data_path: str,
    model_path: str,
    target: str = 'LMN',
    subset: str = 'val',
    features: List[str] = None,
    save_dir: str = 'experiments/shap',
) -> None:
    if features is None:
        features = FEATURES

    logger.info(f'Data path..........: {data_path}')
    logger.info(f'Model path.........: {model_path}')
    logger.info(f'Target.............: {target}')
    logger.info(f'Subset.............: {subset.capitalize()}')
    logger.info(f'Features...........: {features}')
    logger.info(f'Output dir.........: {save_dir}')

    # Read a data frame with predictions
    df = pd.read_excel(data_path)

    # Extract target data frame
    if target == 'LMN' or target == 'STS':
        df_ = df[df['Metric'] == target]
    else:
        raise ValueError(f'Unknown target value {target}')

    # Extract subset data frame
    if subset == 'train':
        df_ = df_[df_['Split'] == 'Training']
    elif subset == 'val':
        df_ = df_[df_['Split'] == 'Validation']
    elif subset == 'all':
        df_ = df_.copy()
    else:
        raise ValueError(f'Unknown subset value {subset}')

    # Initialize model
    model = Regressor(model_path=model_path)

    # Initialize SHAP explainer and calculate Shapley values
    os.makedirs(save_dir, exist_ok=True)
    explainer_path = os.path.join(save_dir, f'SHAP_{target}_{subset}.pickle')
    if Path(explainer_path).exists():
        with open(explainer_path, 'rb') as f:
            shap_values = pickle.load(f)
            logger.info(f'Load SHAP values: {explainer_path}')
    elif not Path(explainer_path).exists():
        data_shap = df_[FEATURES].values
        explainer = shap.Explainer(
            model=model,
            masker=data_shap,
            output_names=[target],
        )
        shap_values = explainer(data_shap)
        shap_values.feature_names = FEATURES
        with open(explainer_path, 'wb') as f:
            pickle.dump(shap_values, f)
            logger.info(f'Save SHAP values: {explainer_path}')

    else:
        raise ValueError('Unexpected error occurred while saving/loading SHAP values')

    # Create and save data frame with SHAP values
    column_names = [f'{col}_shap' for col in FEATURES]
    df_shap = pd.DataFrame(shap_values.values, columns=column_names)
    df_.reset_index(drop=True, inplace=True)
    df_out = pd.concat([df_, df_shap], axis=1)
    save_path_shap = Path(explainer_path).with_suffix('.xlsx')
    df_out.to_excel(
        save_path_shap,
        sheet_name='SHAP',
        index=False,
    )

    # Create a beeswarm plot
    col2num = {col: i for i, col in enumerate(FEATURES)}
    feature_order = list(map(col2num.get, FEATURES))
    fig = plt.figure()
    shap.plots.beeswarm(
        shap_values,
        feature_order=feature_order,
        axis_color='#000000',
        color_bar=False,
        show=False,
    )
    plt.gcf().set_size_inches(5, 5)
    plt.xlabel('SHAP')
    plt.xlim([-0.65, 0.65])
    plt.rc('font', size=0)  # Decrease the padding (not font)
    plt.axhline(-1, linewidth=2, color='black', alpha=1)
    plt.axvline(0, linewidth=1, color='black', alpha=1)
    plt.rcParams['font.family'] = 'BentonSans'

    # Save a beeswarm plot
    save_path_figure = Path(explainer_path).with_suffix('.png')
    fig.tight_layout()
    fig.show()
    fig.savefig(save_path_figure, dpi=500)

    shap_abs_values = list(shap_values.abs.mean(0).values)
    shap_abs_values = [round(x, 4) for x in shap_abs_values]
    logger.info(f'SHAP values........: {shap_abs_values}')
    logger.info(f'Features...........: {shap_values.feature_names}')
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

    parser = argparse.ArgumentParser(description='Compute SHAP values')
    parser.add_argument('--data_path', default='dataset/predictions.xlsx', nargs='+')
    parser.add_argument('--model_path', required=True, type=str)
    parser.add_argument('--target', default='LMN', type=str, choices=['STS', 'LMN'])
    parser.add_argument('--subset', default='val', type=str, choices=['train', 'val', 'all'])
    parser.add_argument('--features', default=FEATURES, nargs='+', type=str)
    parser.add_argument('--save_dir', default='experiments/shap', type=str)
    args = parser.parse_args()

    main(
        data_path=args.data_path,
        model_path=args.model_path,
        target=args.target,
        subset=args.subset,
        features=args.features,
        save_dir=args.save_dir,
    )
