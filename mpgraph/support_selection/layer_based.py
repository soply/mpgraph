# coding: utf8
from timeit import default_timer as timer

import numpy as np
from tabulate import tabulate

from ..mp_utils import approximate_solve_mp_fixed_support

def largest_support_occuring_in_each_layer(tiling, show_table=False,
                                           target_support=None):
    """
    """
    starttime = timer()
    supports = tiling.intersection_support_per_layer()
    largest_support = supports[max(supports.keys())]
    if target_support is not None:
        table = np.zeros((len(supports.keys()), 6))
    else:
        table = np.zeros((len(supports.keys()), 5))
    for i, layer in enumerate(supports.keys()):
        support = supports[layer]
        u_I, v_I = approximate_solve_mp_fixed_support(support, tiling.A,
                                                      tiling.y)
        table[i, 0] = layer
        table[i, 1] = len(support)
        if len(support) > 0:
            table[i, 2] = np.min(np.abs(u_I[support]))  # signal strength
            table[i, 3] = np.max(np.abs(v_I))  # noise level/noise strength
            table[i, 4] = table[i, 2] / table[i, 3] # SNR
        if target_support is not None:
            # Set table[i,5] to symmetric support difference
            table[i, 5] = len(np.setdiff1d(support, target_support)) + \
                len(np.setdiff1d(target_support, support))
    # Sort table
    ranking = np.argsort(table[:, 0])
    table = table[ranking, :]
    if show_table:
        header = ["Layer", "# Active", "c", "d", "SNR", "Symmetric Diff"]
        print tabulate(table, headers=header)
    elapsed_time = timer() - starttime
    return table, largest_support, elapsed_time
