# coding: utf8
from timeit import default_timer as timer

import numpy as np
from tabulate import tabulate

from mp_utils import approximate_solve_mp_fixed_support


def highest_support_constrained_snr(tiling, show_table=False,
                                    target_support=None):
    """ Method to rank the supports connected to a reconstructed tiling by
    calculating the signal to noise ratio solely based on the support. In
    particular, this method calculates approximative solutions u_I and v_I
    by an unpenalised least squares regression on the respective support I for
    u_I and an unpenalised least squares regression of the residual for v_I (not
    restricted to specific support). The signal to noise for a single tiling
    element in the tiling is then given as

    min(|u_i|, i in suppport(tilingelement))/||v_I||_infty (1).

    Parameters
    ----------------
    tiling : object of class Tiling
        Reconstructed tiling from which we take supports that shall be ranked.

    show_table (optional, default False): Boolean True or False
        If true, a summary of the ranking in printed to the terminal.

    target_support (optional, default None): numpy array, shape (n_support)
        If given, the supports of all tiling elements in the tiling will be
        compared to a specific target support (ie. the support with which
        specific data has been created) and the symmetric difference is
        stored into the table.

    Returns
    -----------------
    table: Numpy array, shape (n_tilingelements, 5 or 6)
        The results of the support ranking, columnwise meaning:
            0 : Length of support of tiling element
            1 : Signal strength min(|u_i|, i in suppport(tilingelement))
            2 : Noise strength np.max(np.abs(v_I))
            3 : Signal to noise ratio (entry 1 / entry 2)
            4 : Distance from root element (minimal number of Lasso path steps)
            5 (optional) : Symmetric difference to target support

    best_tilingelement : Tilingelement of the tiling that scores the highest
        signal to noise ratio.

    elapsed_time : Time necessary to perform the ranking.

    Remarks
    -----------------
    Only works well for problems with arbitrary signal noise but low measurement
    noise, ie. in the model A(u + v) = y + epsilon, epsilon must be small if
    this ranking should provide good results.
    """
    starttime = timer()
    tiling_elements = tiling.root_element.bds_order()
    if target_support is not None:
        table = np.zeros((len(tiling_elements.keys()), 6))
    else:
        table = np.zeros((len(tiling_elements.keys()), 5))
    for i, (tilingelement, layer) in enumerate(tiling_elements.iteritems()):
        if layer == 0:
            # Skip root element
            continue
        te_supp = tilingelement.support
        u_I, v_I = approximate_solve_mp_fixed_support(te_supp, tiling.A,
                                                      tiling.y)
        table[i, 0] = len(te_supp)
        table[i, 1] = np.min(np.abs(u_I[te_supp]))  # signal strength
        table[i, 2] = np.max(np.abs(v_I))  # noise level/noise strength
        table[i, 3] = table[i, 1] / table[i, 2]
        table[i, 4] = layer
        if target_support is not None:
            # Set table[i,5] to symmetric support difference
            table[i, 5] = len(np.setdiff1d(te_supp, target_support)) + \
                len(np.setdiff1d(target_support, te_supp))
    # Sort table
    ranking = np.argsort(table[:, 3])
    table = table[ranking, :]
    if show_table:
        header = ["# Active", "c", "d", "SNR", "Layer", "Symmetric Diff"]
        print tabulate(table, headers=header)
    elapsed_time = timer() - starttime
    return table, elapsed_time
