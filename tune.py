import os
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from bayes_opt.event import Events
from bayes_opt.logger import JSONLogger
from bayes_opt import BayesianOptimization

from tools.utils import calculate_design_score

os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%d.%m.%Y %I:%M:%S',
    filename='logs/{:s}.log'.format(Path(__file__).stem),
    filemode='w',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
from tools.model import Regressor


def scoring_function(
        HGT: float,
        DIA: float,
        ANG: float,
        CVT: float,
        THK: float,
        EM: float,
) -> float:

    model_lumen = Regressor(model_path=lumen_model_path)
    model_stress = Regressor(model_path=stress_model_path)

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
        stress_threshold=stress_threshold,
    )

    return score


def main(
        param_bounds: dict,
        acquisition_func: str,
        num_steps: int,
        exploration_points: int,
        kappa: float,
        xi: float,
        seed: int,
        save_dir: str,
) -> None:

    # Log tuning parameters
    logger.info(f'Lumen model............: {lumen_model_path}')
    logger.info(f'Stress model...........: {stress_model_path}')
    logger.info(f'Parameter bounds.......: {param_bounds}')
    logger.info(f'Acquisition function...: {acquisition_func.upper()}')
    logger.info(f'Exploration points.....: {exploration_points}')
    logger.info(f'Optimization steps.....: {num_steps}')
    logger.info(f'Kappa..................: {kappa}')
    logger.info(f'Xi.....................: {xi}')
    logger.info(f'Stress threshold.......: {stress_threshold}')

    optimizer = BayesianOptimization(
        f=scoring_function,
        pbounds=param_bounds,
        verbose=2,
        random_state=seed,
    )

    os.makedirs(save_dir, exist_ok=True)
    json_path = os.path.join(
        save_dir,
        f'tuning_{acquisition_func.upper()}_{num_steps}steps_{exploration_points}points.json'
    )
    json_logger = JSONLogger(path=json_path)
    optimizer.subscribe(Events.OPTIMIZATION_STEP, json_logger)

    optimizer.maximize(
        acq=acquisition_func,
        init_points=exploration_points,
        n_iter=num_steps,
        kappa=kappa,
        xi=xi,
    )
    best_design = optimizer.max

    logger.info('')
    logger.info(f"Best score.............: {best_design['target']:.3f}")
    logger.info(f"Best HGT...............: {best_design['params']['HGT']:.2f}")
    logger.info(f"Best DIA...............: {best_design['params']['DIA']:.2f}")
    logger.info(f"Best ANG...............: {best_design['params']['ANG']:.2f}")
    logger.info(f"Best CVT...............: {best_design['params']['CVT']:.2f}")
    logger.info(f"Best THK...............: {best_design['params']['THK']:.2f}")
    logger.info(f"Best EM................: {best_design['params']['EM']:.2f}")

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
    logger.info('')
    logger.info(f'Tuning output..........: {save_path}')


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
    parser.add_argument('--acquisition_func', default='ucb', type=str, help='ucb, ei, poi')
    parser.add_argument('--num_steps', default=2000, type=int)
    parser.add_argument('--exploration_points', default=100, type=int)
    parser.add_argument('--kappa', default=10, type=float)
    parser.add_argument('--xi', default=0.1, type=float)
    parser.add_argument('--stress_threshold', default=10, type=float)
    parser.add_argument('--seed', default=11, type=int)
    parser.add_argument('--save_dir', default='calculations', type=str)
    args = parser.parse_args()

    global lumen_model_path
    global stress_model_path
    global stress_threshold
    lumen_model_path = args.lumen_model_path
    stress_model_path = args.stress_model_path
    stress_threshold = args.stress_threshold

    main(
        param_bounds=args.param_bounds,
        acquisition_func=args.acquisition_func,
        num_steps=args.num_steps,
        exploration_points=args.exploration_points,
        kappa=args.kappa,
        xi=args.xi,
        seed=args.seed,
        save_dir=args.save_dir,
    )
