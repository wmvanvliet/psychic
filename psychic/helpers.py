import csv, itertools
import numpy as np

def to_one_of_n(labels, class_rows=None):
    '''
    Convert a list with integers to one-of-N coding for to use in a DataSet.
    Note that the rows correspond to the classes in *sorted* order.

    >>> to_one_of_n([0, 0, 0, 2, 0, 1])
    array([[ True,  True,  True, False,  True, False],
           [False, False, False, False, False,  True],
           [False, False, False,  True, False, False]], dtype=bool)
    '''
    a = np.asarray(labels, int)
    if a.ndim != 1:
        raise ValueError('Labels should be 1D')
    if not class_rows:
        class_rows = np.unique(a) # is automatically sorted
    labels = np.zeros((len(class_rows), a.size), dtype=np.bool)
    for i, n in enumerate(class_rows):
        labels[i, a==n] = True
    return labels

def hard_max(data):
    '''
    Find the maximum of each column and return an array containing 1 on the
    location of each maximum. If a column contains a NaN, the output column
    consists of NaNs.
    '''
    data = np.atleast_2d(data)
    assert data.shape[0] != 0
    if data.shape[1] == 0: 
        return data.copy()
    result = np.zeros(data.shape)
    result[np.argmax(data, axis=0),list(range(data.shape[1]))] = 1
    result[:, np.any(np.isnan(data), axis=0)] *= np.nan
    return result

def write_csv_table(rows, fname):
    f = open(fname, 'w')
    csv.writer(f).writerows(rows)
    f.close()

def write_latex_table(rows, fname):
    rows = list(rows)
    ncols = max(len(r) for r in rows)
    f = open(fname, 'w')
    f.write('\\begin{tabular}{%s}\n' % ' '.join('c'*ncols))
    for r in rows:
        f.write(' & '.join(map(str, r)) + '\\\\\n')
    f.write('\\end{tabular}\n')
    f.close()

