import os
import time
import logging
import warnings
import argparse
from typing import List
from pathlib import Path
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
from math import sqrt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer, PowerTransformer

os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%d.%m.%Y %I:%M:%S',
    filename='logs/{:s}.log'.format(Path(__file__).stem),
    filemode='w',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
from supervised.automl import AutoML


def main(
    data_path: str,
    target: str,
    scale_features: bool,
    scale_target: bool,
    mode: str,
    metric: str,
    algorithms: List[str],
    seed: int,
    save_dir: str,
):

    # Log main parameters
    logger.info('')
    logger.info(f'Data path......: {data_path}')
    logger.info(f'Target.........: {target}')
    logger.info(f'Scale features.: {scale_features}')
    logger.info(f'Scale target...: {scale_target}')
    logger.info(f'Mode...........: {mode}')
    logger.info(f'Metric.........: {metric}')
    logger.info(f'Algorithms.....: {algorithms}')
    logger.info(f'Seed...........: {seed}')

    # Read source data
    df = pd.read_excel(data_path)
    features = [
        'H',
        'D',
        'F',
        'R',
        'T',
    ]
    X = df[features]
    y = df[target]

    # Preprocess data
    if scale_features:
        # scaler_input = StandardScaler().fit(X)
        # scaler_input = QuantileTransformer().fit(X)
        # scaler_input = PowerTransformer().fit(X)
        scaler_features = RobustScaler().fit(X)
        X = pd.DataFrame(scaler_features.transform(X), columns=features)

    if scale_target:
        # scaler_output = StandardScaler().fit(y.to_frame())
        # scaler_output = QuantileTransformer().fit(y.to_frame())
        # scaler_output = RobustScaler().fit(y.to_frame())
        # scaler_output = PowerTransformer(method='yeo-johnson').fit(y.to_frame())
        scaler_target = RobustScaler().fit(y.to_frame())
        y = pd.Series(scaler_target.transform(y.to_frame()).squeeze())

    validation_strategy = {
        'validation_type': 'kfold',
        'k_folds': 10,
        'shuffle': True,
        'stratify': True,
        'random_seed': seed,
    }

    t = time.localtime()
    current_time = time.strftime('%H%M%S_%d%m', t)
    experiment_path = os.path.join(save_dir, f'{target}_{mode.lower()}_{metric}_{current_time}')
    logger.info(f'Directory......: {experiment_path}')

    automl = AutoML(
        results_path=experiment_path,
        ml_task='regression',
        mode=mode,
        eval_metric=metric,
        algorithms=algorithms,
        random_state=seed,
        validation_strategy=validation_strategy,
        total_time_limit=3600,
        optuna_time_budget=3600,
    )

    automl.fit(X, y)
    automl.report()

    # Make predictions
    y_pred = automl.predict(X)
    y_pred = pd.Series(y_pred)
    mae_val = mean_absolute_error(y_true=y.squeeze(), y_pred=y_pred)
    mse_val = mean_squared_error(y_true=y.squeeze(), y_pred=y_pred)
    rmse_val = sqrt(mean_squared_error(y_true=y.squeeze(), y_pred=y_pred))
    r2_val = r2_score(y_true=y.squeeze(), y_pred=y_pred)
    mape_val = mean_absolute_percentage_error(y_true=y.squeeze(), y_pred=y_pred)

    logger.info(f'Metrics........:')
    logger.info(f'MAE............: {mae_val:.3f}')
    logger.info(f'RMSE...........: {rmse_val:.3f}')
    logger.info(f'MSE............: {mse_val:.3f}')
    logger.info(f'R2.............: {r2_val:.3f}')
    logger.info(f'MAPE...........: {mape_val:.2%}')

    # Save dataframe
    if scale_features:
        X = scaler_features.inverse_transform(X)
        X = pd.DataFrame(X, columns=features)

    if scale_target:
        y = scaler_target.inverse_transform(y.to_frame())
        y = pd.DataFrame(y, columns=[target])

    y.reset_index(drop=True, inplace=True)
    y_pred.reset_index(drop=True, inplace=True)
    df_out = pd.concat(
        [
            X,
            y.squeeze().rename('{:s}_gt'.format(target)),
            y_pred.rename('{:s}_pred'.format(target))
        ],
        axis=1,
    )
    df_out['Residual'] = df_out['{:s}_gt'.format(target)] - df_out['{:s}_pred'.format(target)]
    df_out.to_excel(
        f'{experiment_path}/predictions.xlsx',
        sheet_name='Predictions',
        index=True,
        index_label='Design',
        startrow=0,
        startcol=0,
    )


if __name__ == '__main__':

    ALGORITHMS = [
        'Baseline',
        'Decision Tree',
        'Random Forest',
        'Extra Trees',
        'LightGBM',
        'Xgboost',
        'CatBoost',
        'Neural Network',
        'Nearest Neighbors',
        ]
    
    parser = argparse.ArgumentParser(description='Dataset conversion')
    parser.add_argument('--data_path', default='dataset/data.xlsx', type=str)
    parser.add_argument('--target', default='VMS', type=str, help='Lumen or VMS')
    parser.add_argument('--scale_features', action='store_true')
    parser.add_argument('--scale_target', action='store_true')
    parser.add_argument('--mode', default='Compete', type=str)
    parser.add_argument('--metric', default='mae', type=str)
    parser.add_argument('--algorithms', default=ALGORITHMS, nargs='+', type=str)
    parser.add_argument('--seed', default=11, type=int)
    parser.add_argument('--save_dir', default='experiments', type=str)
    args = parser.parse_args()

    main(
        data_path=args.data_path,
        target=args.target,
        scale_features=args.scale_features,
        scale_target=args.scale_target,
        mode=args.mode,
        metric=args.metric,
        algorithms=args.algorithms,
        seed=args.seed,
        save_dir=args.save_dir,
    )

    logger.info('')
    logger.info('Model training complete')
