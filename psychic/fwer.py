'''
Module implementing some methods to control family-wise error rate when
performing multiple comparison tests. Implemented are:

    1. Bonferroni
    2. Bonferrino-Holm
    3. Benjamini-Hochberg
'''

import numpy as np

def _sort(x):
    '''
    Sort data and also return a reverse index
    '''
    n = len(x)
    o = np.argsort(x)
    original_order = np.zeros(n, dtype=int)
    original_order[o] = np.arange(n)
    return (np.asarray(x)[o], original_order)

def bonferroni(ps):
    '''
    Adjust p-values from multiple comparisons using Bonferroni correction.

    :param ps: List of p values
    '''
    n = len(ps)
    return np.clip(np.asarray(ps) * n, 0, 1)

def bonferroni_holm(ps):
    '''
    Adjust p-values from multiple comparisons using Bonferroni-Holm correction.

    :param ps: List of p values
    '''
    n = len(ps)
    ps, original_order = _sort(ps)
    adj_ps = np.asarray(ps) * (n - np.arange(n))
    return np.clip(adj_ps[original_order], 0, 1)

def benjamini_hochberg(ps):
    '''
    Adjust p-values from multiple comparisons using Benjamini-Hochberg
    correction.

    :param ps: List of p values
    '''
    n = len(ps)
    ps, original_order = _sort(ps)
    adj_ps = (np.asarray(ps) * n) / (np.arange(n) + 1.0)
    return np.clip(adj_ps[original_order], 0, 1)
