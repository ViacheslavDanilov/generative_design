import os
import json
import timeit
import logging
import argparse
from pathlib import Path
from datetime import datetime

import optuna
import numpy as np
import pandas as pd

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


def objective(
        trial: optuna.Trial,
        model_lumen: Regressor,
        model_stress: Regressor,
        param_bounds: dict,
        materials: dict,
        json_path: str,
):

    HGT = trial.suggest_float(
        'HGT',
        param_bounds['HGT'][0],
        param_bounds['HGT'][1],
    )
    DIA = trial.suggest_float(
        'DIA',
        param_bounds['DIA'][0],
        param_bounds['DIA'][1],
    )
    ANG = trial.suggest_float(
        'ANG',
        param_bounds['ANG'][0],
        param_bounds['ANG'][1],
    )
    CVT = trial.suggest_float(
        'CVT',
        param_bounds['CVT'][0],
        param_bounds['CVT'][1],
    )
    THK = trial.suggest_float(
        'THK',
        param_bounds['THK'][0],
        param_bounds['THK'][1],
    )
    MTL = trial.suggest_int(
        'MTL',
        param_bounds['MTL'][0],
        param_bounds['MTL'][1],
        step=1,
    )
    ELM = materials[f'material_{MTL}']['ELM']
    UTS = materials[f'material_{MTL}']['UTS']

    start = timeit.default_timer()
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
    log_string = f'Iteration: {(trial.number + 1):04d} - ' \
                 f'Elapsed: {int(stop - start):3d} - ' \
                 f'HGT: {HGT:4.1f} - ' \
                 f'DIA: {DIA:4.1f} - ' \
                 f'ANG: {ANG:5.1f} - ' \
                 f'CVT: {CVT:4.1f} - ' \
                 f'THK: {THK:4.2f} - ' \
                 f'ELM: {ELM:4.1f} - ' \
                 f'UTS: {UTS:4.1f} - ' \
                 f'MTL: {MTL:d} - ' \
                 f'Lumen: {lumen:4.2f} - ' \
                 f'Stress: {stress:5.2f} - ' \
                 f'Score: {score:5.3f}'
    logger.info(log_string)

    # Logging
    now = datetime.now()
    log_params = {
        'Datetime': now.strftime('%d-%m-%Y %H:%M:%S'),
        'Iteration': trial.number + 1,
        'Elapsed': stop - start,
        'HGT': HGT,
        'DIA': DIA,
        'ANG': ANG,
        'CVT': CVT,
        'THK': THK,
        'ELM': ELM,
        'UTS': UTS,
        'MTL': MTL,
        'Lumen': lumen,
        'Stress': stress,
        'Score': score,
    }

    with open(json_path, mode='a', encoding='utf-8') as outfile:
        outfile.write(json.dumps(log_params))
        outfile.write("\n")
        outfile.close()

    return score


def main(
        lumen_model_path: str,
        stress_model_path: str,
        sampler: str,
        pruner: str,
        param_bounds: dict,
        materials_path: str,
        num_trials: int,
        seed: int,
        save_dir: str,
) -> None:

    os.makedirs(save_dir, exist_ok=True)

    json_path = os.path.join(
        save_dir,
        f'{sampler}_{pruner}.json'
    )
    try:
        os.remove(json_path)
    except OSError:
        pass
    open(json_path, 'w')

    model_lumen = Regressor(model_path=lumen_model_path)
    model_stress = Regressor(model_path=stress_model_path)

    f = open(materials_path, 'r')
    materials = json.loads(f.read())

    # Log tuning parameters
    logger.info(f'Lumen model............: {lumen_model_path}')
    logger.info(f'Stress model...........: {stress_model_path}')
    logger.info(f'Sampler................: {sampler}')
    logger.info(f'Pruner.................: {pruner}')
    logger.info(f'Parameter bounds.......: {param_bounds}')
    logger.info(f'Number of materials....: {len(materials)}')
    logger.info(f'Number of trials.......: {num_trials}')
    logger.info(f'Seed...................: {seed}')
    logger.info(f'Output directory.......: {save_dir}')
    logger.info('')

    # Sampling method
    if sampler == 'RS':
        sampler = optuna.samplers.RandomSampler(seed=seed)
    elif sampler == 'CMA':
        sampler = optuna.samplers.CmaEsSampler(seed=seed)
    elif sampler == 'TPE':
        sampler = optuna.samplers.TPESampler(seed=seed)
    elif sampler == 'NSGA':
        sampler = optuna.samplers.NSGAIISampler(seed=seed)
    elif sampler == 'MOTPE':
        sampler = optuna.samplers.MOTPESampler(seed=seed)
    elif sampler == 'QMC':
        sampler = optuna.samplers.QMCSampler(seed=seed)
    else:
        raise ValueError(f'Sampler {sampler} is not available.')

    # Pruning method
    if pruner == 'Halving':
        pruner = optuna.pruners.SuccessiveHalvingPruner()
    elif pruner == 'Hyperband':
        pruner = optuna.pruners.HyperbandPruner()
    elif pruner == 'Median':
        pruner = optuna.pruners.MedianPruner()
    else:
        raise ValueError(f'Pruner {pruner} is not available.')

    # Create and fit a study
    objective_func = lambda trial: objective(
        trial=trial,
        model_lumen=model_lumen,
        model_stress=model_stress,
        param_bounds=param_bounds,
        materials=materials,
        json_path=json_path,
    )
    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        pruner=pruner,
    )
    study.optimize(
        func=objective_func,
        n_trials=num_trials,
        show_progress_bar=True,
    )

    # Compute and save feature importance values
    importance = {}
    importance['fanova'] = optuna.importance.get_param_importances(
        study,
        evaluator=optuna.importance.FanovaImportanceEvaluator(
            n_trees=64,
            max_depth=64,
            seed=seed,
        ),
        normalize=True,
    )
    importance['mdi'] = optuna.importance.get_param_importances(
        study,
        evaluator=optuna.importance.MeanDecreaseImpurityImportanceEvaluator(
            n_trees=64,
            max_depth=64,
            seed=seed,
        ),
        normalize=False,
    )
    df_fanova = pd.DataFrame.from_dict(
        importance['fanova'],
        orient='index',
        columns=['fANOVA'],
    )
    df_mdi = pd.DataFrame.from_dict(
        importance['mdi'],
        orient='index',
        columns=['MDI'],
    )
    df_importance = pd.concat([df_fanova, df_mdi], axis='columns')
    _save_path = Path(json_path).with_suffix('')
    save_path = Path(f'{_save_path}_importance').with_suffix('.xlsx')
    df_importance.to_excel(
        save_path,
        sheet_name='Importance',
        index=True,
    )

    # Create column with best scores and save data frame
    df = pd.read_json(json_path, lines=True)
    df['Score (max)'] = np.nan
    best_val = -np.inf
    for idx, row in df.iterrows():
        score = row['Score']
        if score > best_val:
            df.at[idx, 'Score (max)'] = score
            best_val = score
        else:
            df.at[idx, 'Score (max)'] = best_val
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
        'MTL': (1, 20),             # material index
    }

    parser = argparse.ArgumentParser(description='Hyperparameter optimization using Optuna')
    parser.add_argument('--lumen_model_path', required=True, type=str)
    parser.add_argument('--stress_model_path', required=True, type=str)
    parser.add_argument('--sampler', default='TPE', type=str, choices=['RS', 'CMA', 'TPE', 'NSGA', 'MOTPE', 'QMC'])
    parser.add_argument('--pruner', default='Halving', type=str, choices=['Median', 'Halving', 'Hyperband'])
    parser.add_argument('--param_bounds', default=BOUNDS, type=str)
    parser.add_argument('--materials_path', default='dataset/materials.json', type=str)
    parser.add_argument('--num_trials', default=10000, type=int)
    parser.add_argument('--seed', default=11, type=int)
    parser.add_argument('--save_dir', default='experiments/tune', type=str)
    args = parser.parse_args()

    main(
        lumen_model_path=args.lumen_model_path,
        stress_model_path=args.stress_model_path,
        sampler=args.sampler,
        pruner=args.pruner,
        param_bounds=args.param_bounds,
        materials_path=args.materials_path,
        num_trials=args.num_trials,
        seed=args.seed,
        save_dir=args.save_dir,
    )
