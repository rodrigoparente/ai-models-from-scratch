# third-party imports
import numpy as np
from tabulate import tabulate


def shuffle(X, y):
    assert len(X) == len(y)
    p = np.random.permutation(len(X))
    return X[p], y[p]


def display_cm(cm):
    print(tabulate(cm, tablefmt='psql', floatfmt=".2f", colalign=('center',),
                   showindex='always', headers=[i for i in range(cm.shape[1])]))
