import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from bayes_opt.event import Events
from bayes_opt.logger import JSONLogger
from bayes_opt import BayesianOptimization

from tools.model import Regressor
from tools.utils import calculate_design_score


# TODO: remove later
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

    # TODO: load models once + pass as a function parameter
    model_lumen = Regressor(model_path='experiments/LMN_compete_mae_Raw_MinMax_Source_1928_2504')
    model_stress = Regressor(model_path='experiments/Smax_compete_mae_Robust_Power_Source_2011_2004')

    _data_point = np.array(
        [
            HGT,
            DIA,
            ANG,
            CVT,
            THK,
            EM,
        ]
    )
    data_point = _data_point.reshape(1, -1)

    lumen = float(model_lumen(data_point))
    stress = float(model_stress(data_point))

    score = calculate_design_score(
        lumen_abs=lumen,
        stress_abs=stress,
        stress_threshold=stress_threshold,              # TODO: pass as a function parameter
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

    optimizer = BayesianOptimization(
        f=scoring_function,            # FIXME: call with stress_threshold and model paths
        pbounds=param_bounds,
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

    _df = pd.read_json(json_path, lines=True)
    df = pd.concat(
        [
            _df.datetime.apply(pd.Series),
            _df.params.apply(pd.Series),
            _df['target'],
        ],
        axis=1,
    )
    df.rename(
        columns={
            'datetime': 'Datetime',
            'elapsed': 'Elapsed',
            'delta': 'Delta',
            'target': 'Score',
        },
        inplace=True
    )

    save_path = Path(json_path).with_suffix('.xlsx')
    df.to_excel(
        save_path,
        sheet_name='Optimization',
        index=True,
        index_label='Iteration',
        startrow=0,
        startcol=0,
    )

    os.remove(json_path)


if __name__ == '__main__':

    BOUNDS = {
        'HGT': (10, 25),
        'DIA': (15, 40),
        'ANG': (-30, 30),
        'CVT': (0, 100),
        'THK': (0.1, 1.0),
        'EM':  (0.5, 20),
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
