import os
import time
import logging
import warnings
import argparse
from pickle import dump
from typing import List
from pathlib import Path
warnings.simplefilter(action='ignore', category=FutureWarning)

import shutil
import numpy as np
import pandas as pd
from math import sqrt
from sklearn.metrics import *
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, PowerTransformer

from tools.utils import get_golden_features, calc_mape

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
    mode: str,
    target: str,
    features: List[str],
    test_size: float,
    golden_features_path: str,
    feature_scale: str,
    target_scale: str,
    metric: str,
    algorithms: List[str],
    seed: int,
    save_dir: str,
):

    if golden_features_path is not None:
        assert os.path.isfile(golden_features_path), 'JSON file with golden features is not found'

    t = time.localtime()
    current_time = time.strftime('%H%M_%d%m', t)
    if golden_features_path is not None:
        experiment_path = os.path.join(
            save_dir,
            f"{target}_{mode.lower()}_{metric}_{feature_scale}_{target_scale}_{'Golden'}_{current_time}"
        )
    else:
        experiment_path = os.path.join(
            save_dir,
            f"{target}_{mode.lower()}_{metric}_{feature_scale}_{target_scale}_{'Source'}_{current_time}"
        )
    os.makedirs(experiment_path, exist_ok=True)

    # Read source data
    df = pd.read_excel(data_path)
    X = df[features].values
    y = df[target].values
    y = y.reshape(-1, 1)

    # Preprocess features
    if feature_scale == 'MinMax':
        input_scaler = MinMaxScaler().fit(X)
    if feature_scale == 'Standard':
        input_scaler = StandardScaler().fit(X)
    if feature_scale == 'Robust':
        input_scaler = RobustScaler().fit(X)
    if feature_scale == 'Power':
        input_scaler = PowerTransformer().fit(X)

    X = input_scaler.transform(X) if feature_scale != 'Raw' else X

    # Add golden features
    if golden_features_path is not None:
        X = get_golden_features(
            input_data=X,
            golden_features_path=golden_features_path,
        )

    # Preprocess target
    if target_scale == 'MinMax':
        output_scaler = MinMaxScaler().fit(y)
    if target_scale == 'Standard':
        output_scaler = StandardScaler().fit(y)
    if target_scale == 'Robust':
        output_scaler = RobustScaler().fit(y)
    if target_scale == 'Power':
        output_scaler = PowerTransformer(method='yeo-johnson').fit(y)

    y = output_scaler.transform(y) if target_scale != 'Raw' else y

    validation_strategy = {
        'validation_type': 'custom',
    }
    train_idx, val_idx = train_test_split(
        np.arange(len(X)),
        test_size=test_size,
        random_state=seed,
        shuffle=True,
    )
    cv = [(train_idx, val_idx)]

    # Log main parameters
    logger.info(f'Data path..........: {data_path}')
    logger.info(f'Target.............: {target}')
    logger.info(f'Scale features.....: {feature_scale}')
    logger.info(f'Scale target.......: {target_scale}')
    logger.info(f'Train/Val ratio....: {100*(1-test_size):.0f}/{100*test_size:.0f}')
    logger.info(f'Mode...............: {mode}')
    logger.info(f'Metric.............: {metric}')
    logger.info(f'Algorithms.........: {algorithms}')
    logger.info(f'Seed...............: {seed}')
    logger.info(f'Directory..........: {experiment_path}')
    logger.info(f'Golden features....: {golden_features_path}')

    # Train models
    automl = AutoML(
        results_path=experiment_path,
        ml_task='regression',
        mode=mode,
        eval_metric=metric,
        algorithms=algorithms,
        random_state=seed,
        validation_strategy=validation_strategy,
        total_time_limit=3600,
        optuna_time_budget=180,
        explain_level=2,
    )
    automl.fit(
        X=X,
        y=y.squeeze(),
        cv=cv,
    )
    automl.report()

    # Save input data scaler
    try:
        dump(input_scaler, open(os.path.join(experiment_path, 'input_scaler.pkl'), 'wb'))
    except Exception as e:
        pass

    # Save output data scaler
    try:
        dump(output_scaler, open(os.path.join(experiment_path, 'output_scaler.pkl'), 'wb'))
    except Exception as e:
        pass

    # Make predictions
    y_pred = automl.predict(X)
    y_pred = y_pred.reshape(-1, 1)

    # Scale back the data to the original representation
    X = X[:, :len(features)]
    if feature_scale != 'Raw':
        X = input_scaler.inverse_transform(X)

    if target_scale != 'Raw':
        y = output_scaler.inverse_transform(y)
        y_pred = output_scaler.inverse_transform(y_pred)

    if np.any(np.isnan(y_pred)):
        nan_mask = np.isnan(y_pred).squeeze()
        nan_idx = np.where(nan_mask == True)[0]
        _train_idx = [i for i in train_idx if i not in nan_idx]
        train_idx = np.array(_train_idx)
        _val_idx = [i for i in val_idx if i not in nan_idx]
        val_idx = np.array(_val_idx)
        logger.info(f'Found and dropped {nan_mask.sum()} NaNs in predictions')

    # Compute all metrics
    mae_train = mean_absolute_error(y_true=np.take(y, train_idx), y_pred=np.take(y_pred, train_idx))
    mae_val = mean_absolute_error(y_true=np.take(y, val_idx), y_pred=np.take(y_pred, val_idx))

    mse_train = mean_squared_error(y_true=np.take(y, train_idx), y_pred=np.take(y_pred, train_idx))
    mse_val = mean_squared_error(y_true=np.take(y, val_idx), y_pred=np.take(y_pred, val_idx))

    rmse_train = sqrt(mean_squared_error(y_true=np.take(y, train_idx), y_pred=np.take(y_pred, train_idx)))
    rmse_val = sqrt(mean_squared_error(y_true=np.take(y, val_idx), y_pred=np.take(y_pred, val_idx)))

    r2_train = r2_score(y_true=np.take(y, train_idx), y_pred=np.take(y_pred, train_idx))
    r2_val = r2_score(y_true=np.take(y, val_idx), y_pred=np.take(y_pred, val_idx))

    mape_train = calc_mape(y_true=np.take(y, train_idx), y_pred=np.take(y_pred, train_idx))
    mape_val = calc_mape(y_true=np.take(y, val_idx), y_pred=np.take(y_pred, val_idx))

    pearson_train, _ = pearsonr(x=np.take(y, train_idx).squeeze(), y=np.take(y_pred, train_idx).squeeze())
    pearson_val, _ = pearsonr(x=np.take(y, val_idx).squeeze(), y=np.take(y_pred, val_idx).squeeze())

    logger.info('')
    logger.info(f'Metrics............: Train / Val')
    logger.info(f'MAPE...............: {mape_train:.2} / {mape_val:.2}')
    logger.info(f'MAE................: {mae_train:.3f} / {mae_val:.3f}')
    logger.info(f'RMSE...............: {rmse_train:.3f} / {rmse_val:.3f}')
    logger.info(f'MSE................: {mse_train:.3f} / {mse_val:.3f}')
    logger.info(f'R2.................: {r2_train:.3f} / {r2_val:.3f}')
    logger.info(f'Pearson............: {pearson_train:.2%} / {pearson_val:.2%}')

    # Save metrics dataframe
    metrics = {
        'Metric': ['MAPE', 'MAE', 'RMSE', 'MSE', 'R2', 'Pearson'],
        'Training': [mape_train, mae_train, rmse_train, mse_train, r2_train, pearson_train],
        'Validation': [mape_val, mae_val, rmse_val, mse_val, r2_val, pearson_val],
    }
    df_metrics = pd.DataFrame(metrics)
    df_metrics.to_excel(
        f'{experiment_path}/metrics.xlsx',
        sheet_name='Metrics',
        index=True,
        index_label='ID',
        startrow=0,
        startcol=0,
    )

    data = np.hstack([X, y, y_pred])
    df_out = pd.DataFrame(data, columns=[*features, f'{target}_gt', f'{target}_pred'])
    df_out['Residual'] = df_out['{:s}_gt'.format(target)] - df_out['{:s}_pred'.format(target)]
    df_out['Error'] = df_out['Residual'].abs()
    df_out['Subset'] = 'Training'
    df_out.loc[val_idx, 'Subset'] = 'Validation'
    df_out.to_excel(
        f'{experiment_path}/predictions.xlsx',
        sheet_name='Predictions',
        index=True,
        index_label='Design',
        startrow=0,
        startcol=0,
    )

    logger.info('')
    logger.info('Model training complete')

    if golden_features_path is not None:
        shutil.copy(golden_features_path, os.path.join(experiment_path, 'new_features.json'))
    shutil.copy(f'logs/{Path(__file__).stem}.log', experiment_path)


if __name__ == '__main__':

    ALGORITHMS = [
        'Baseline',
        'Linear',
        'Decision Tree',
        'Random Forest',
        'Extra Trees',
        'Neural Network',
        'LightGBM',
        'Xgboost',
        'CatBoost',
        ]

    FEATURES = [
        'HGT',
        'DIA',
        'ANG',
        'CVT',
        'THK',
        'EM',
    ]

    parser = argparse.ArgumentParser(description='Dataset conversion')
    parser.add_argument('--data_path', default='dataset/data.xlsx', type=str)
    parser.add_argument('--mode', default='Explain', type=str)
    parser.add_argument('--target', default='Smax', type=str, help='LMN, VMS, Smax, LEmax')
    parser.add_argument('--features', default=FEATURES, nargs='+', type=str)
    parser.add_argument('--test_size', default=0.2, type=float, help='size of the test split')
    parser.add_argument('--golden_features_path', default=None, type=str)
    parser.add_argument('--feature_scale', default='Robust', type=str, help='Raw, MinMax, Standard, Robust, Power')
    parser.add_argument('--target_scale', default='Power', type=str, help='Raw, MinMax, Standard, Robust, Power')
    parser.add_argument('--metric', default='mae', type=str)
    parser.add_argument('--algorithms', default=ALGORITHMS, nargs='+', type=str)
    parser.add_argument('--seed', default=11, type=int)
    parser.add_argument('--save_dir', default='experiments', type=str)
    args = parser.parse_args()

    main(
        data_path=args.data_path,
        mode=args.mode,
        target=args.target,
        features=args.features,
        test_size=args.test_size,
        golden_features_path=args.golden_features_path,
        feature_scale=args.feature_scale,
        target_scale=args.target_scale,
        metric=args.metric,
        algorithms=args.algorithms,
        seed=args.seed,
        save_dir=args.save_dir,
    )
