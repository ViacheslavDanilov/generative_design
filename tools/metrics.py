from typing import Tuple

import numpy as np

EPSILON = 1e-10


def _error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """Simple error"""
    return y_true - y_pred


def _percentage_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """Percentage error

    Note: result is NOT multiplied by 100
    """
    return _error(y_true, y_pred) / (y_true + EPSILON)


def _naive_forecasting(
    y_true: np.ndarray,
    seasonality: int = 1,
) -> np.ndarray:
    """Naive forecasting method which just repeats previous samples"""
    return y_true[:-seasonality]


def _relative_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    benchmark: np.ndarray = None,
) -> np.ndarray:
    """Relative Error."""
    if benchmark is None or isinstance(benchmark, int):
        # If no benchmark prediction provided - use naive forecasting
        if not isinstance(benchmark, int):
            seasonality = 1
        else:
            seasonality = benchmark
        return _error(y_true[seasonality:], y_pred[seasonality:]) / (
            _error(y_true[seasonality:], _naive_forecasting(y_true, seasonality)) + EPSILON
        )

    return _error(y_true, y_pred) / (_error(y_true, benchmark) + EPSILON)


def _bounded_relative_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    benchmark: np.ndarray = None,
) -> np.ndarray:
    """Bounded Relative Error"""
    if benchmark is None or isinstance(benchmark, int):
        # If no benchmark prediction provided - use naive forecasting
        if not isinstance(benchmark, int):
            seasonality = 1
        else:
            seasonality = benchmark

        abs_err = np.abs(_error(y_true[seasonality:], y_pred[seasonality:]))
        abs_err_bench = np.abs(
            _error(y_true[seasonality:], _naive_forecasting(y_true, seasonality)),
        )
    else:
        abs_err = np.abs(_error(y_true, y_pred))
        abs_err_bench = np.abs(_error(y_true, benchmark))

    return abs_err / (abs_err + abs_err_bench + EPSILON)


def _geometric_mean(a, axis=0, dtype=None) -> np.ndarray:
    """Geometric mean."""
    if not isinstance(a, np.ndarray):  # if not an ndarray object attempt to convert it
        log_a = np.log(np.array(a, dtype=dtype))
    elif dtype:  # Must change the default dtype allowing array type
        if isinstance(a, np.ma.MaskedArray):
            log_a = np.log(np.ma.asarray(a, dtype=dtype))
        else:
            log_a = np.log(np.asarray(a, dtype=dtype))
    else:
        log_a = np.log(a)
    return np.exp(log_a.mean(axis=axis))


def mse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """Mean Squared Error"""
    return np.mean(np.square(_error(y_true, y_pred)))


def rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """Root Mean Squared Error"""
    return np.sqrt(mse(y_true, y_pred))


def nrmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """Normalized Root Mean Squared Error"""
    return rmse(y_true, y_pred) / (y_true.max() - y_true.min())


def me(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """Mean Error."""
    return np.mean(_error(y_true, y_pred))


def mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """Mean Absolute Error."""
    return np.mean(np.abs(_error(y_true, y_pred)))


mad = mae  # Mean Absolute Deviation (it is the same as MAE)


def gmae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """Geometric Mean Absolute Error"""
    return _geometric_mean(np.abs(_error(y_true, y_pred)))


def mdae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """Median Absolute Error"""
    return np.median(np.abs(_error(y_true, y_pred)))


def mpe(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """Mean Percentage Error"""
    return np.mean(_percentage_error(y_true, y_pred))


def mape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """Mean Absolute Percentage Error

    Properties:
        + Easy to interpret
        + Scale independent
        - Biased, not symmetric
        - Undefined when y_true[t] == 0

    Note: result is NOT multiplied by 100
    """
    return np.mean(np.abs(_percentage_error(y_true, y_pred)))


def mdape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """Median Absolute Percentage Error

    Note: result is NOT multiplied by 100
    """
    return np.median(np.abs(_percentage_error(y_true, y_pred)))


def smape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """Symmetric Mean Absolute Percentage Error

    Note: result is NOT multiplied by 100
    """
    return np.mean(2.0 * np.abs(y_true - y_pred) / ((np.abs(y_true) + np.abs(y_pred)) + EPSILON))


def smdape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """Symmetric Median Absolute Percentage Error

    Note: result is NOT multiplied by 100
    """
    return np.median(2.0 * np.abs(y_true - y_pred) / ((np.abs(y_true) + np.abs(y_pred)) + EPSILON))


def maape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """Mean Arctangent Absolute Percentage Error

    Note: result is NOT multiplied by 100
    """
    return np.mean(np.arctan(np.abs((y_true - y_pred) / (y_true + EPSILON))))


def mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    seasonality: int = 1,
) -> np.ndarray:
    """Mean Absolute Scaled Error

    Baseline (benchmark) is computed with naive forecasting (shifted by @seasonality)
    """
    return mae(y_true, y_pred) / mae(y_true[seasonality:], _naive_forecasting(y_true, seasonality))


def std_ae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """Normalized Absolute Error"""
    _mae = mae(y_true, y_pred)
    return np.sqrt(np.sum(np.square(_error(y_true, y_pred) - _mae)) / (len(y_true) - 1))


def std_ape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """Normalized Absolute Percentage Error"""
    _mape = mape(y_true, y_pred)
    return np.sqrt(np.sum(np.square(_percentage_error(y_true, y_pred) - _mape)) / (len(y_true) - 1))


def rmspe(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """Root Mean Squared Percentage Error

    Note: result is NOT multiplied by 100
    """
    return np.sqrt(np.mean(np.square(_percentage_error(y_true, y_pred))))


def rmdspe(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """Root Median Squared Percentage Error

    Note: result is NOT multiplied by 100
    """
    return np.sqrt(np.median(np.square(_percentage_error(y_true, y_pred))))


def rmsse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    seasonality: int = 1,
) -> np.ndarray:
    """Root Mean Squared Scaled Error"""
    q = np.abs(_error(y_true, y_pred)) / mae(
        y_true[seasonality:],
        _naive_forecasting(y_true, seasonality),
    )
    return np.sqrt(np.mean(np.square(q)))


def inrse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """Integral Normalized Root Squared Error"""
    return np.sqrt(
        np.sum(np.square(_error(y_true, y_pred))) / np.sum(np.square(y_true - np.mean(y_true))),
    )


def rrse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """Root Relative Squared Error"""
    return np.sqrt(np.sum(np.square(y_true - y_pred)) / np.sum(np.square(y_true - np.mean(y_true))))


def mre(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    benchmark: np.ndarray = None,
) -> np.ndarray:
    """Mean Relative Error"""
    return np.mean(_relative_error(y_true, y_pred, benchmark))


def rae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """Relative Absolute Error (aka Approximation Error)"""
    return np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true - np.mean(y_true))) + EPSILON)


def mrae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    benchmark: np.ndarray = None,
) -> np.ndarray:
    """Mean Relative Absolute Error"""
    return np.mean(np.abs(_relative_error(y_true, y_pred, benchmark)))


def mdrae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    benchmark: np.ndarray = None,
) -> np.ndarray:
    """Median Relative Absolute Error"""
    return np.median(np.abs(_relative_error(y_true, y_pred, benchmark)))


def gmrae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    benchmark: np.ndarray = None,
) -> np.ndarray:
    """Geometric Mean Relative Absolute Error"""
    return _geometric_mean(np.abs(_relative_error(y_true, y_pred, benchmark)))


def mbrae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    benchmark: np.ndarray = None,
) -> np.ndarray:
    """Mean Bounded Relative Absolute Error"""
    return np.mean(_bounded_relative_error(y_true, y_pred, benchmark))


def umbrae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    benchmark: np.ndarray = None,
) -> np.ndarray:
    """Unscaled Mean Bounded Relative Absolute Error"""
    __mbrae = mbrae(y_true, y_pred, benchmark)
    return __mbrae / (1 - __mbrae)


def mda(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """Mean Directional Accuracy"""
    return np.mean(
        (np.sign(y_true[1:] - y_true[:-1]) == np.sign(y_pred[1:] - y_pred[:-1])).astype(int),
    )


def wape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """Weighted Absolute Percentage Error"""
    return np.sum(np.abs(y_true - y_pred)) / np.sum(y_true)


def r_squared(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """Coefficient of determination or R^2"""
    return 1 - np.sum(np.square(_error(y_true, y_pred))) / np.sum(
        np.square(y_true - np.mean(y_true)),
    )


def pearson(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """Pearson's R"""
    return np.corrcoef(y_true, y_pred)[0, 1]


METRICS = {
    'MAPE': mape,
    'WAPE': wape,
    'MAE': mae,
    'MAAPE': maape,
    'MASE': mase,
    'MSE': mse,
    'RMSE': rmse,
    'NRMSE': nrmse,
    'R^2': r_squared,
    'Pearson': pearson,
    'MBRAE': mbrae,
    'UMBRAE': umbrae,
    'ME': me,
    'MAD': mad,
    'GMAE': gmae,
    'MDAE': mdae,
    'MPE': mpe,
    'MDAPE': mdape,
    'SMAPE': smape,
    'SMDAPE': smdape,
    'STDAE': std_ae,
    'STDAPE': std_ape,
    'RMSPE': rmspe,
    'RMDSPE': rmdspe,
    'RMSSE': rmsse,
    'INRSE': inrse,
    'RRSE': rrse,
    'MRE': mre,
    'RAE': rae,
    'MRAE': mrae,
    'MDRAE': mdrae,
    'GMRAE': gmrae,
    'MDA': mda,
}


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: Tuple[str, ...] = (
        'MAPE',
        'WAPE',
        'MAE',
        'MAAPE',
        'MASE',
        'MSE',
        'RMSE',
        'NRMSE',
        'R^2',
        'Pearson',
    ),
) -> dict:
    """Evaluate the performance of a model specifying the metrics to be computed"""
    null_idx = np.where(y_true == 0)[0]
    if null_idx.size > 0:
        y_true = np.delete(y_true, null_idx, axis=0)
        y_pred = np.delete(y_pred, null_idx, axis=0)

    results = {}
    for name in metrics:
        try:
            results[name] = float(METRICS[name](y_true, y_pred))
        except Exception as err:
            results[name] = np.nan
            print(f'Unable to compute metric {name}: {err}')
    return results


def calculate_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """Evaluate the performance of a model computing all metrics"""
    return calculate_metrics(y_true, y_pred, metrics=tuple(METRICS.keys()))


if __name__ == '__main__':
    y_true = np.array([34, 37, 44, 47, 48, 48, 46, 43, 32, 27, 26, 24])
    y_pred = np.array([37, 40, 46, 44, 46, 50, 45, 44, 34, 30, 22, 23])

    m = calculate_metrics(y_true, y_pred)
    m_all = calculate_all_metrics(y_true, y_pred)
