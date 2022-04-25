import os
import time
import logging
import argparse
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd


def main(
        lumen_model_path: str,
        stress_model_path: str,
        save_dir: str,
) -> None:

    return None


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Hyperparameter optimization')
    parser.add_argument('--lumen_model_path', default=None, type=str)
    parser.add_argument('--stress_model_path', default=None, type=str)
    parser.add_argument('--save_dir', default='calculations', type=str)
    args = parser.parse_args()

    main(
        lumen_model_path=args.lumen_model_path,
        stress_model_path=args.stress_model_path,
        save_dir=args.save_dir,
    )
