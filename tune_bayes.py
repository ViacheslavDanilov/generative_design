import os
import json
import timeit
import logging
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization
from bayes_opt import SequentialDomainReductionTransformer as SDRTransformer

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


def get_material_idx(
        MTL: float,
        materials: dict,
) -> int:
    idx = int(MTL - 1e-10) if MTL == len(materials) + 1 else int(MTL)
    return idx


def scoring_function(
        HGT: float,
        DIA: float,
        ANG: float,
        CVT: float,
        THK: float,
        MTL: float,
) -> float:

    global iteration
    global materials
    global lumen_model_path
    global stress_model_path

    idx = get_material_idx(
        MTL=MTL,
        materials=materials,
    )
    assert type(idx) == int, 'MTL index must be an integer value'
    ELM = materials[f'material_{idx}']['ELM']
    UTS = materials[f'material_{idx}']['UTS']

    start = timeit.default_timer()
    model_lumen = Regressor(model_path=lumen_model_path)
    model_stress = Regressor(model_path=stress_model_path)

    _data_point = np.array(
        [
            HGT,
            DIA,
            ANG,
            CVT,
            THK,
            ELM,
        ]
    )
    data_point = _data_point.reshape(1, -1)

    lumen = float(model_lumen(data_point))
    lumen = 0 if lumen < 0 else lumen

    stress = float(model_stress(data_point))
    stress = 0 if stress < 0 else stress

    score = calculate_design_score(
        lumen_abs=lumen,
        stress_abs=stress,
        uts=UTS,
    )
    stop = timeit.default_timer()

    # Reporting
    log_string = f'Iteration: {(iteration + 1):04d} - ' \
                 f'Elapsed: {int(stop - start):3d} - ' \
                 f'HGT: {HGT:4.1f} - ' \
                 f'DIA: {DIA:4.1f} - ' \
                 f'ANG: {ANG:5.1f} - ' \
                 f'CVT: {CVT:4.1f} - ' \
                 f'THK: {THK:4.2f} - ' \
                 f'ELM: {ELM:4.1f} - ' \
                 f'UTS: {UTS:4.1f} - ' \
                 f'MTL: {idx:d} - ' \
                 f'Lumen: {lumen:4.2f} - ' \
                 f'Stress: {stress:5.2f} - ' \
                 f'Score: {score:5.3f}'
    logger.info(log_string)

    # Logging
    now = datetime.now()
    log_params = {
        'Datetime': now.strftime('%d-%m-%Y %H:%M:%S'),
        'Iteration': iteration + 1,
        'Elapsed': stop - start,
        'HGT': HGT,
        'DIA': DIA,
        'ANG': ANG,
        'CVT': CVT,
        'THK': THK,
        'ELM': ELM,
        'UTS': UTS,
        'MTL': idx,
        'Lumen': lumen,
        'Stress': stress,
        'Score': score,
    }

    with open(json_path, mode='a', encoding='utf-8') as outfile:
        outfile.write(json.dumps(log_params))
        outfile.write("\n")
        outfile.close()

    iteration += 1

    return score


def main(
        param_bounds: dict,
        use_sdr: bool,
        sdr_shrinkage: float,
        sdr_pan: float,
        sdr_zoom: float,
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
    logger.info(f'Use SDR transformer....: {use_sdr}')
    logger.info(f'SDR shrinkage..........: {sdr_shrinkage}')
    logger.info(f'SDR pan................: {sdr_pan}')
    logger.info(f'SDR zoom...............: {sdr_zoom}')
    logger.info(f'Acquisition function...: {acquisition_func.upper()}')
    logger.info(f'Exploration points.....: {exploration_points}')
    logger.info(f'Optimization steps.....: {num_steps}')
    logger.info(f'Kappa..................: {kappa}')
    logger.info(f'Xi.....................: {xi}')
    logger.info('')

    if use_sdr:
        bounds_transformer = SDRTransformer(
            gamma_osc=sdr_shrinkage,
            gamma_pan=sdr_pan,
            eta=sdr_zoom,
        )
        json_suffix = '_SDR'
    else:
        bounds_transformer = None
        json_suffix = ''

    optimizer = BayesianOptimization(
        f=scoring_function,
        pbounds=param_bounds,
        bounds_transformer=bounds_transformer,
        verbose=2,
        random_state=seed,
    )

    os.makedirs(save_dir, exist_ok=True)
    global json_path
    json_path = os.path.join(
        save_dir,
        f'tuning_{acquisition_func.upper()}_{num_steps}steps_{exploration_points}points{json_suffix}.json'
    )

    try:
        os.remove(json_path)
    except OSError:
        pass
    open(json_path, 'w')

    optimizer.maximize(
        acq=acquisition_func,
        init_points=exploration_points,
        n_iter=num_steps,
        kappa=kappa,
        xi=xi,
    )
    best_design = optimizer.max

    idx = get_material_idx(
        MTL=best_design['params']['MTL'],
        materials=materials,
    )

    logger.info('')
    logger.info(f'Best design')
    logger.info(f"Score..................: {best_design['target']:.3f}")
    logger.info(f"HGT....................: {best_design['params']['HGT']:.2f}")
    logger.info(f"DIA....................: {best_design['params']['DIA']:.2f}")
    logger.info(f"ANG....................: {best_design['params']['ANG']:.2f}")
    logger.info(f"CVT....................: {best_design['params']['CVT']:.2f}")
    logger.info(f"THK....................: {best_design['params']['THK']:.2f}")
    logger.info(f"ELM....................: {materials[f'material_{idx}']['ELM']:.2f}")
    logger.info(f"UTS....................: {materials[f'material_{idx}']['UTS']:.2f}")
    logger.info(f"MTL....................: {idx:d}")

    df = pd.read_json(json_path, lines=True)
    save_path = Path(json_path).with_suffix('.xlsx')
    df.to_excel(
        save_path,
        sheet_name='Optimization',
        index=False,
    )
    logger.info('')
    logger.info(f'Tuning output..........: {save_path}')


if __name__ == '__main__':

    BOUNDS = {
        'HGT': (10, 25),            # valve height
        'DIA': (19, 33),            # valve diameter
        'ANG': (-30, 30),           # free edge angle
        'CVT': (0.0, 1.0),          # leaflet curvature
        'THK': (0.3, 1.0),          # leaflet thickness
        'MTL': (1, 21),             # material index
    }

    parser = argparse.ArgumentParser(description='Hyperparameter optimization')
    parser.add_argument('--lumen_model_path', default=None, type=str)
    parser.add_argument('--stress_model_path', default=None, type=str)
    parser.add_argument('--param_bounds', default=BOUNDS, type=str)
    parser.add_argument('--materials_path', default='dataset/materials.json', type=str)
    parser.add_argument('--use_sdr', action='store_true')
    parser.add_argument('--sdr_shrinkage', default=0.95, type=float)
    parser.add_argument('--sdr_pan', default=1.0, type=float)
    parser.add_argument('--sdr_zoom', default=0.99, type=float)
    parser.add_argument('--acquisition_func', default='ucb', type=str, help='ucb, ei, poi')
    parser.add_argument('--num_steps', default=2000, type=int)
    parser.add_argument('--exploration_points', default=250, type=int)
    parser.add_argument('--kappa', default=10, type=float)
    parser.add_argument('--xi', default=0.1, type=float)
    parser.add_argument('--seed', default=11, type=int)
    parser.add_argument('--save_dir', default='calculations/tuning', type=str)
    args = parser.parse_args()

    iteration = 0
    f = open(args.materials_path, 'r')
    materials = json.loads(f.read())
    lumen_model_path = args.lumen_model_path
    stress_model_path = args.stress_model_path

    main(
        param_bounds=args.param_bounds,
        use_sdr=args.use_sdr,
        sdr_shrinkage=args.sdr_shrinkage,
        sdr_pan=args.sdr_pan,
        sdr_zoom=args.sdr_zoom,
        acquisition_func=args.acquisition_func,
        num_steps=args.num_steps,
        exploration_points=args.exploration_points,
        kappa=args.kappa,
        xi=args.xi,
        seed=args.seed,
        save_dir=args.save_dir,
    )
