import os
from warnings import simplefilter
simplefilter('ignore', UserWarning)
simplefilter('ignore', FutureWarning)

import numpy as np
from pickle import load
from supervised.automl import AutoML


class Regressor:
    def __init__(
            self,
            model_path: str,
    ):

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

        # Predict data samples
        _output = self.model.predict(_data)

        # Scale output data
        if self.output_scaler is not None:
            output = self.output_scaler.inverse_transform(_output.reshape(-1, 1))
        else:
            output = _output.reshape(-1, 1)

        return output
