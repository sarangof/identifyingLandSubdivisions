import numpy as np
import pandas as pd

"""
Here, 0 will be irregular and 1 will be subdivision
"""

def standardize_metric_1(series):
    return series

def standardize_metric_2(series):
    result = np.where(series > 100, 100, series)
    return pd.Series(1 - (result / 100), index=series.index)

def standardize_metric_3(series):
    series = np.where(series > 40, 40, series)
    return series/40

def standardize_metric_4(series):
    return series

def standardize_metric_5(series):
    series = np.where(series > 324, 324, series)
    return series/324.

def standardize_metric_6(series):
    return series

def standardize_metric_7(series):
    series = np.where(series < 30, 30, series)
    series = np.where(series > 200, 200, series)
    return (1.-((series - 30.)/170.))

def standardize_metric_8(series):
    series = np.where(series < 0.85, 0.85, series)
    return series

def standardize_metric_9(series):
    return 1-(series/90)

def standardize_metric_10(series):
    series = np.where(series > 4000, 4000, series)
    return series/4000.

def standardize_metric_11(series):
    return series

def standardize_metric_12(series):
    series = np.where(series > 200, 200, series)
    return series/200.

# Map metrics to their respective functions
standardization_functions = {
    'metric_1': standardize_metric_1,
    'metric_2': standardize_metric_2,
    'metric_3': standardize_metric_3,
    'metric_4': standardize_metric_4,
    'metric_5': standardize_metric_5,
    'metric_6': standardize_metric_6,
    'metric_7': standardize_metric_7,
    'metric_8': standardize_metric_8,
    'metric_9': standardize_metric_9,
    'metric_10': standardize_metric_10,
    'metric_11': standardize_metric_11,
    'metric_12': standardize_metric_12
}