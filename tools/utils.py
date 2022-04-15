import re
import json

import numpy as np

'''A function that takes 'Rel_VMS' and 'Rel_Lumen' as input and 
   return total efficiency score, normalized in the range 0-1.'''
def efficiency_score(Rel_Area, Rel_VMS):
    Relative_Area_function = lambda x: 2 / (1 + np.exp((1 - x) * 10))
    Stress_funtion = lambda x: 1 - x if (x < 1) else 0

    Relative_Area_function_list = [Relative_Area_function(item) for item in Rel_Area]
    Stress_funtion_list = [Stress_funtion(item) for item in Rel_VMS]
    Efficiency_functiuon_score = [(1 - a) * b for a, b in zip(Stress_funtion_list, Relative_Area_function_list)]

    return Efficiency_functiuon_score


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

if __name__ == '__main__':

    c =