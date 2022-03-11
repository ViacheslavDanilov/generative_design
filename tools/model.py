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
            self.output_scaler = load(open(os.path.join(model_path, 'output_scaler.pkl'), 'rb'))
        except FileNotFoundError:
            self.input_scaler = None
            self.output_scaler = None

    def __call__(self, data):

        _data = self.input_scaler.transform(data)
        output_norm = self.model.predict(_data)
        output = self.output_scaler.inverse_transform(output_norm.reshape(-1, 1))

        return output
