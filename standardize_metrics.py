import numpy as np

"""
Here, 0 will be irregular and 1 will be subdivision
"""

def standardize_metric_1(series):
    return series

def standardize_metric_2(series):
    series = np.where(series > 100, 100, series)
    return 1 - (series/100)

def standardize_metric_3(series):
    series = np.where(series > 40, 40, series)
    return series/40

def standardize_metric_4(series):
    return series

def standardize_metric_5(series):
    series = np.where(series > 324, 324, series)
    return series/324

def standardize_metric_6(series):
    return series

def standardize_metric_7(series):
    series = np.where(series < 30, 30, series)
    series = np.where(series > 200, 200, series)
    return (1.-((series - 30)/170))

def standardize_metric_8(series):
    """
    M8 = IF(B/A>=S/T, 1, 1-(B/A)/(S/T). 
    If, for example, (B/A)/(S/T) = 0.1, then M8=0.9
    """
    series = np.where(series > 1, 1, series)
    return (1 - series)

def standardize_metric_9(series):
    return series

def standardize_metric_10(series):
    return 1-(series/90)

def standardize_metric_11(series):
    series = np.where(series > 200, 200, series)
    return series/200

def standardize_metric_12(series):
    return series

def standardize_metric_13(series):
    series = np.where(series > 200, 200, series)
    return series/200

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
    'metric_12': standardize_metric_12,
    'metric_13': standardize_metric_13
}