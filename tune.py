import os
import time
import logging
import argparse
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd
from bayes_opt.event import Events
from bayes_opt.logger import JSONLogger
from bayes_opt import BayesianOptimization

from tools.utils import calculate_design_score


def scoring_function_temp(
        lumen: float,
        stress: float,
) -> float:

    score = lumen ** 2 - (stress + 1) ** 2 + 1

    return score


def scoring_function(
        HGT: float,
        DIA: float,
        ANG: float,
        CVT: float,
        THK: float,
        EM: float,
        stress_threshold: float = 10,
) -> float:

    # Run model 1
    lumen = 0.5             # TODO: get output from model 1

    # Run model 2
    stress = 2.5            # TODO: get output from model 2

    score = calculate_design_score(
        lumen_abs=lumen,
        stress_abs=stress,
        stress_threshold=stress_threshold,
    )

    return score


def main(
        lumen_model_path: str,
        stress_model_path: str,
        param_bounds: dict,
        exploration_strategy: str,
        exploration_points: int,
        num_steps: int,
        stress_threshold: float,
        seed: int,
        save_dir: str,
) -> None:

    # TODO: remove after debugging
    param_bounds_temp = {
        'lumen': (2, 7),
        'stress': (-5, 5),
    }

    optimizer = BayesianOptimization(
        f=scoring_function_temp,            # FIXME: change to normal version + call with stress_threshold
        pbounds=param_bounds_temp,          # FIXME: change to normal version
        verbose=2,
        random_state=seed,
    )

    os.makedirs(save_dir, exist_ok=True)
    json_path = os.path.join(save_dir, 'tuning.json')
    logger = JSONLogger(path=json_path)
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    optimizer.maximize(
        acq=exploration_strategy,
        init_points=exploration_points,
        n_iter=num_steps,
    )

    # TODO: convert tuning.json to tuning.csv and then save it


if __name__ == '__main__':

    BOUNDS = {
        'HGT': (10, 25),
        'DIA': (15, 40),
        'ANG': (-30, 30),
        'CVT': (0, 100),
        'THK': (0.1, 1.0),
        'EM':  (2, 4),          # FIXME: find out the real bounds
    }

    parser = argparse.ArgumentParser(description='Hyperparameter optimization')
    parser.add_argument('--lumen_model_path', default=None, type=str)
    parser.add_argument('--stress_model_path', default=None, type=str)
    parser.add_argument('--param_bounds', default=BOUNDS, type=str)
    parser.add_argument('--exploration_strategy', default='ucb', type=str, help='ucb, ei, poi')
    parser.add_argument('--exploration_points', default=5, type=int)
    parser.add_argument('--num_steps', default=25, type=int)
    parser.add_argument('--stress_threshold', default=10, type=float)
    parser.add_argument('--seed', default=11, type=int)
    parser.add_argument('--save_dir', default='calculations', type=str)
    args = parser.parse_args()

    main(
        lumen_model_path=args.lumen_model_path,
        stress_model_path=args.stress_model_path,
        param_bounds=args.param_bounds,
        exploration_strategy=args.exploration_strategy,
        exploration_points=args.exploration_points,
        num_steps=args.num_steps,
        stress_threshold=args.stress_threshold,
        seed=args.seed,
        save_dir=args.save_dir,
    )
