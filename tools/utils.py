import json
import os
import re
from pathlib import Path
from typing import List, Tuple, Union

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
    alpha: float = 1.0,
) -> Tuple[float, float, float]:
    if not np.isnan(lumen_abs) and not np.isnan(stress_abs):
        lumen_score = calculate_lumen_score(
            lumen_abs=lumen_abs,
        )

        stress_score = calculate_stress_score(
            stress_abs=stress_abs,
            uts=uts,
        )

        # alpha is chosen such that lumen_score is considered alpha times as important as stress_score
        design_score_numer = (1 + alpha**2) * stress_score * lumen_score
        design_score_denom = alpha**2 * stress_score + lumen_score
        design_score = design_score_numer / design_score_denom

    else:
        lumen_score = 0.0
        stress_score = 0.0
        design_score = 0.0

    return lumen_score, stress_score, design_score


def get_file_list(
    src_dirs: Union[List[str], str],
    ext_list: Union[List[str], str],
    file_template: str = '',
) -> List[str]:
    """Get list of files with the specified extensions

    Args:
        src_dirs: directory(s) with files inside
        ext_list: extension(s) used for a search
        file_template: include files with this template
    Returns:
        all_files: a list of file paths
    """
    all_files = []
    src_dirs = [src_dirs] if isinstance(src_dirs, str) else src_dirs
    ext_list = [ext_list] if isinstance(ext_list, str) else ext_list
    for src_dir in src_dirs:
        for root, dirs, files in os.walk(src_dir):
            for file in files:
                file_ext = Path(file).suffix
                file_ext = file_ext.lower()
                if file_ext in ext_list and file_template in file:
                    file_path = os.path.join(root, file)
                    all_files.append(file_path)
    all_files.sort()
    return all_files


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
            _feature_data = np.expand_dims(
                input_data[:, feature_1_id] + input_data[:, feature_2_id],
                axis=1,
            )
        elif new_feature['operation'] == 'diff':
            _feature_data = np.expand_dims(
                input_data[:, feature_1_id] - input_data[:, feature_2_id],
                axis=1,
            )
        elif new_feature['operation'] == 'multiply':
            _feature_data = np.expand_dims(
                input_data[:, feature_1_id] * input_data[:, feature_2_id],
                axis=1,
            )
        elif new_feature['operation'] == 'ratio':
            _feature_data = np.expand_dims(
                input_data[:, feature_1_id] / input_data[:, feature_2_id],
                axis=1,
            )
        else:
            raise ValueError(f"Unknown operation {new_feature['operation']}")

        output_data = np.hstack([output_data, _feature_data])

    return output_data
