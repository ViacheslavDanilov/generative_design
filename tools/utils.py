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
