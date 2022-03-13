import os

from pickle import load
from supervised.automl import AutoML


class Regressor:
    def __init__(
            self,
            model_path: str,
    ):

        self.model = AutoML(results_path=model_path)

        try:
            self.input_scaler = load(open(os.path.join(model_path, 'input_scaler.pkl'), 'rb'))
        except FileNotFoundError:
            self.input_scaler = None

        try:
            self.output_scaler = load(open(os.path.join(model_path, 'output_scaler.pkl'), 'rb'))
        except FileNotFoundError:
            self.output_scaler = None

    def __call__(
            self,
            data
    ):

        # TODO: Add support of golden features
        if self.input_scaler is not None:
            _data = self.input_scaler.transform(data)
        else:
            _data = data.copy()

        _output = self.model.predict(_data)

        if self.output_scaler is not None:
            output = self.output_scaler.inverse_transform(_output.reshape(-1, 1))
        else:
            output = _output.reshape(-1, 1)

        return output
