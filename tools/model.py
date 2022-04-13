import os

import numpy as np
from pickle import load
from supervised.automl import AutoML

from tools.utils import get_golden_features


class Regressor:
    def __init__(
            self,
            model_path: str,
            use_golden_features: bool = False,
    ):

        # Check the existence of the golden_features.json
        if use_golden_features:
            golden_features_path = os.path.join(model_path, 'golden_features.json')
            assert os.path.isfile(golden_features_path), 'JSON file with golden features is not found'
            self.use_golden_features = True
            self.golden_features_path = golden_features_path
        else:
            self.use_golden_features = False
            self.golden_features_path = None

        # Check the existence of the input_scaler.pkl
        try:
            self.input_scaler = load(open(os.path.join(model_path, 'input_scaler.pkl'), 'rb'))
        except FileNotFoundError:
            self.input_scaler = None

        # Check the existence of the output_scaler.pkl
        try:
            self.output_scaler = load(open(os.path.join(model_path, 'output_scaler.pkl'), 'rb'))
        except FileNotFoundError:
            self.output_scaler = None

        # Load the model
        self.model = AutoML(results_path=model_path)

    def __call__(
            self,
            data: np.ndarray,
    ):

        # Scale input data
        if self.input_scaler is not None:
            _data = self.input_scaler.transform(data)
        else:
            _data = data.copy()

        # Use golden features along with the initial _data
        if self.use_golden_features:
            _data = get_golden_features(
                input_data=_data,
                golden_features_path=self.golden_features_path,
            )

        # Predict data samples
        _output = self.model.predict(_data)

        # Scale output data
        if self.output_scaler is not None:
            output = self.output_scaler.inverse_transform(_output.reshape(-1, 1))
        else:
            output = _output.reshape(-1, 1)

        return output
