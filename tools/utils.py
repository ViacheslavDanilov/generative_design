import re
import json

import numpy as np


def calculate_lumen_score(
        lumen_abs: float,
) -> float:
    lumen_score = lumen_abs
    return lumen_score


def calculate_stress_score(
        stress_abs: float,
        uts: float = 8.9,
) -> float:
    stress_rel = stress_abs / uts
    stress_score = 1 - stress_rel if stress_rel < 1 else 0
    return stress_score


def calculate_design_score(
        lumen_abs: float,
        stress_abs: float,
        uts: float,
) -> float:

    if (
            not np.isnan(lumen_abs)
            and not np.isnan(stress_abs)
    ):

        lumen_score = calculate_lumen_score(
            lumen_abs=lumen_abs,
        )

        stress_score = calculate_stress_score(
            stress_abs=stress_abs,
            uts=uts,
        )

        design_score = np.sqrt(lumen_score * stress_score)

    else:
        design_score = 0

    return design_score


def get_golden_features(
        input_data: np.ndarray,
        golden_features_path: str,
) -> np.ndarray:
    f = open(golden_features_path)
    _golden_features = json.load(f)
    golden_features = _golden_features['new_features']

    output_data = np.array(input_data, copy=True)
    for new_feature in golden_features:
        feature_1_id = int(re.findall(r'\d+', new_feature['feature1'])[0]) - 1
        feature_2_id = int(re.findall(r'\d+', new_feature['feature2'])[0]) - 1

        if new_feature['operation'] == 'sum':
            _feature_data = np.expand_dims(input_data[:, feature_1_id] + input_data[:, feature_2_id], axis=1)
        elif new_feature['operation'] == 'diff':
            _feature_data = np.expand_dims(input_data[:, feature_1_id] - input_data[:, feature_2_id], axis=1)
        elif new_feature['operation'] == 'multiply':
            _feature_data = np.expand_dims(input_data[:, feature_1_id] * input_data[:, feature_2_id], axis=1)
        elif new_feature['operation'] == 'ratio':
            _feature_data = np.expand_dims(input_data[:, feature_1_id] / input_data[:, feature_2_id], axis=1)
        else:
            raise ValueError(f"Unknown operation {new_feature['operation']}")

        output_data = np.hstack([output_data, _feature_data])

    return output_data


def calculate_mape(
        y_true: np.ndarray,
        y_pred: np.ndarray,
) -> float:

    null_idx = np.where(y_true == 0)[0]
    if null_idx.size > 0:
        y_true = np.delete(y_true, null_idx, axis=0)
        y_pred = np.delete(y_pred, null_idx, axis=0)

    _mape = np.mean(np.abs((y_true - y_pred) / y_true))
    mape = float(_mape)
    return mape


def calculate_wape(
        y_true: np.ndarray,
        y_pred: np.ndarray,
) -> float:

    null_idx = np.where(y_true == 0)[0]
    if null_idx.size > 0:
        y_true = np.delete(y_true, null_idx, axis=0)
        y_pred = np.delete(y_pred, null_idx, axis=0)

    _wape = np.sum(np.abs(y_true - y_pred)) / np.sum(y_true)
    wape = float(_wape)
    return wape


# TODO: validate and if needed remove this function
def calculate_wmape(
        y_true: np.ndarray,
        y_pred: np.ndarray,
) -> float:

    null_idx = np.where(y_true == 0)[0]
    if null_idx.size > 0:
        y_true = np.delete(y_true, null_idx, axis=0)
        y_pred = np.delete(y_pred, null_idx, axis=0)

    weighted_error = (np.abs(y_true - y_pred) / np.abs(y_true)) * y_true
    _wmape = np.sum(weighted_error) / np.sum(y_true)
    wmape = float(_wmape)
    return wmape


if __name__ == '__main__':

    y_true = np.array([34, 37, 44, 47, 48, 48, 46, 43, 32, 27, 26, 24])
    y_pred = np.array([37, 40, 46, 44, 46, 50, 45, 44, 34, 30, 22, 23])

    print(calculate_mape(y_true, y_pred))
    print(calculate_wape(y_true, y_pred))
    print(calculate_wmape(y_true, y_pred))
