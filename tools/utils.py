import re
import json

import numpy as np


def score_func(
        rel_area: np.ndarray = 0,
        stress: np.ndarray = 0,
        stress_max: float = 0,
) -> np.ndarray:
    rel_stress = stress / stress_max
    relative_area_func = lambda a: 2 / (1 + np.exp((1 - a) * 10))
    stress_func = lambda s: 1-s if s < 1 else 0
    stress_score = np.ndarray(0)
    for i in rel_stress:
        stress_score = np.append(stress_score, stress_func(i))
    stress_score = stress_score.reshape(len(stress_score), 1)
    area_score = np.nan_to_num(relative_area_func(rel_area))
    score = stress_score * area_score
    return score


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
