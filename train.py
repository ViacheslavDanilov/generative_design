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
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, PowerTransformer

from tools.metrics import *
from tools.utils import get_golden_features

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
    val_strategy: str,
    k_folds: int,
    val_size: float,
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
    temp = val_strategy.upper() if val_strategy == 'cv' else val_strategy.capitalize()
    experiment_path = os.path.join(
        save_dir,
        f"{target}_{mode.lower()}_{metric}_{feature_scale}_{target_scale}_{temp}_{current_time}"
    )
    os.makedirs(experiment_path, exist_ok=True)
    del temp

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

    if val_strategy == 'split':
        validation_strategy = {
            'validation_type': 'custom',
        }
        train_idx, val_idx = train_test_split(
            np.arange(len(X)),
            test_size=val_size,
            random_state=seed,
            shuffle=True,
        )
        cv = [(train_idx, val_idx)]
    elif val_strategy == 'cv':
        validation_strategy = {
            'validation_type': 'kfold',
            'k_folds': k_folds,
            'shuffle': True,
            'stratify': True,
            'random_seed': seed,
        }
        cv = None
    elif val_strategy == 'auto':
        validation_strategy = 'auto'
        cv = None
    else:
        raise ValueError(f'Unknown validation strategy: {val_strategy}')

    # Log main parameters
    val_str = val_strategy.upper() if val_strategy == 'cv' else val_strategy.capitalize()
    folds_str = k_folds if (val_strategy == 'cv' and val_strategy != 'auto') else None
    ratio_str = f'{(1 - val_size):.2f}/{val_size:.2f}' if (val_strategy == 'split' and val_strategy != 'auto') else None
    logger.info(f'Data path..........: {data_path}')
    logger.info(f'Target.............: {target}')
    logger.info(f'Scale features.....: {feature_scale}')
    logger.info(f'Scale target.......: {target_scale}')
    logger.info(f'Validation strategy: {val_str}')
    logger.info(f'CV folds...........: {folds_str}')
    logger.info(f'Train/Val ratio....: {ratio_str}')
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
        total_time_limit=18000,
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

    # Load indices
    folds_dir = os.path.join(experiment_path, 'folds')
    k_folds = len(os.listdir(folds_dir)) // 2
    train_idx = {}
    val_idx = {}
    for fold_idx in range(k_folds):
        _train_indices_path = os.path.join(folds_dir, f'fold_{fold_idx}_train_indices.npy')
        _val_indices_path = os.path.join(folds_dir, f'fold_{fold_idx}_validation_indices.npy')
        train_idx[f'fold {fold_idx+1}'] = np.load(_train_indices_path)
        val_idx[f'fold {fold_idx+1}'] = np.load(_val_indices_path)

    # Remove NaNs from computations
    if np.any(np.isnan(y_pred)):
        nan_mask = np.isnan(y_pred).squeeze()
        nan_idx = np.where(nan_mask == True)[0]
        for fold_idx in range(k_folds):
            _train_idx = [i for i in train_idx[f'fold {fold_idx+1}'] if i not in nan_idx]
            train_idx[f'fold {fold_idx+1}'] = np.array(_train_idx)
            _val_idx = [i for i in val_idx[f'fold {fold_idx+1}'] if i not in nan_idx]
            val_idx[f'fold {fold_idx+1}'] = np.array(_val_idx)
        logger.info(f'Found and dropped {nan_mask.sum()} NaNs in predictions')

    # Compute all metrics
    df_metrics_train = pd.DataFrame()
    df_metrics_val = pd.DataFrame()
    for fold_idx in range(k_folds):

        _metrics_train = calculate_all_metrics(
            y_true=np.take(y, train_idx[f'fold {fold_idx+1}']),
            y_pred=np.take(y_pred, train_idx[f'fold {fold_idx+1}']),
        )
        _metrics_val = calculate_all_metrics(
            y_true=np.take(y, val_idx[f'fold {fold_idx+1}']),
            y_pred=np.take(y_pred, val_idx[f'fold {fold_idx+1}']),
        )

        df_metrics_train = df_metrics_train.append(_metrics_train, ignore_index=True)
        df_metrics_val = df_metrics_val.append(_metrics_val, ignore_index=True)

    # Save metrics dataframe
    writer = pd.ExcelWriter(f'{experiment_path}/metrics.xlsx', engine='xlsxwriter')
    df_metrics_train.index += 1
    df_metrics_train.to_excel(
        writer,
        sheet_name='Train',
        index=True,
        index_label='Fold',
    )
    df_metrics_val.index += 1
    df_metrics_val.to_excel(
        writer,
        sheet_name='Val',
        index=True,
        index_label='Fold',
    )
    writer.save()

    # Save predictions
    data = np.hstack([X, y, y_pred])
    df_out = pd.DataFrame(data, columns=[*features, f'{target}_gt', f'{target}_pred'])
    df_out['Residual'] = df_out['{:s}_gt'.format(target)] - df_out['{:s}_pred'.format(target)]
    df_out['Error'] = df_out['Residual'].abs()

    for fold_idx in range(k_folds):
        df_out[f'Fold {fold_idx+1}'] = 'Training'
        df_out.loc[val_idx[f'fold {fold_idx+1}'], f'Fold {fold_idx+1}'] = 'Validation'

    df_out.to_excel(
        f'{experiment_path}/predictions.xlsx',
        sheet_name='Predictions',
        index=True,
        index_label='Design',
        startrow=0,
        startcol=0,
    )

    logger.info('')
    logger.info(f'Metrics............: Train / Val')
    logger.info(f'MAPE...............: {df_metrics_train["MAPE"].mean():.3f} / {df_metrics_val["MAPE"].mean():.3f}')
    logger.info(f'WAPE...............: {df_metrics_train["WAPE"].mean():.3f} / {df_metrics_val["WAPE"].mean():.3f}')
    logger.info(f'MAE................: {df_metrics_train["MAE"].mean():.3f} / {df_metrics_val["MAE"].mean():.3f}')
    logger.info(f'MAAPE..............: {df_metrics_train["MAAPE"].mean():.3f} / {df_metrics_val["MAAPE"].mean():.3f}')
    logger.info(f'MASE...............: {df_metrics_train["MASE"].mean():.3f} / {df_metrics_val["MASE"].mean():.3f}')
    logger.info(f'MSE................: {df_metrics_train["MSE"].mean():.3f} / {df_metrics_val["MSE"].mean():.3f}')
    logger.info(f'RMSE...............: {df_metrics_train["RMSE"].mean():.3f} / {df_metrics_val["RMSE"].mean():.3f}')
    logger.info(f'NRMSE..............: {df_metrics_train["NRMSE"].mean():.3f} / {df_metrics_val["NRMSE"].mean():.3f}')
    logger.info(f'R^2................: {df_metrics_train["R^2"].mean():.3f} / {df_metrics_val["R^2"].mean():.3f}')
    logger.info(f'Pearson............: {df_metrics_train["Pearson"].mean():.2%} / {df_metrics_val["Pearson"].mean():.2%}')
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
        'ELM',
    ]

    parser = argparse.ArgumentParser(description='Dataset conversion')
    parser.add_argument('--data_path', default='dataset/data.xlsx', type=str)
    parser.add_argument('--mode', default='Explain', type=str)
    parser.add_argument('--target', default='Smax', type=str, choices=['Smax', 'LMN', 'VMS', 'LEmax'])
    parser.add_argument('--features', default=FEATURES, nargs='+', type=str)
    parser.add_argument('--val_strategy', default='split', type=str, choices=['cv', 'split', 'auto'])
    parser.add_argument('--k_folds', default=5, type=int, help='Number of cross-validation folds')
    parser.add_argument('--val_size', default=0.2, type=float, help='size of the test split')
    parser.add_argument('--golden_features_path', default=None, type=str)
    parser.add_argument('--feature_scale', default='MinMax', type=str, choices=['Raw', 'MinMax', 'Standard', 'Robust', 'Power'])
    parser.add_argument('--target_scale', default='Power', type=str, choices=['Raw', 'MinMax', 'Standard', 'Robust', 'Power'])
    parser.add_argument('--metric', default='rmse', type=str, choices=['mse', 'rmse', 'mae'])
    parser.add_argument('--algorithms', default=ALGORITHMS, nargs='+', type=str)
    parser.add_argument('--seed', default=11, type=int)
    parser.add_argument('--save_dir', default='experiments', type=str)
    args = parser.parse_args()

    main(
        data_path=args.data_path,
        mode=args.mode,
        target=args.target,
        features=args.features,
        val_strategy=args.val_strategy,
        k_folds=args.k_folds,
        val_size=args.val_size,
        golden_features_path=args.golden_features_path,
        feature_scale=args.feature_scale,
        target_scale=args.target_scale,
        metric=args.metric,
        algorithms=args.algorithms,
        seed=args.seed,
        save_dir=args.save_dir,
    )
