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
    table: Numpy array, shape (n_tilingelements, 6 or 7)
        The results of the support ranking, columnwise meaning:
            0 : Unique identifier of the tiling element in the complete tiling.
            1 : Length of support of tiling element
            2 : Signal strength min(|u_i|, i in suppport(tilingelement))
            3 : Noise strength np.max(np.abs(v_I))
            4 : Signal to noise ratio (entry 1 / entry 2)
            5 : Distance from root element (minimal number of Lasso path steps)
            6 (optional) : Symmetric difference to target support

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
    # Remove root element
    del tiling_elements[tiling.root_element]
    if target_support is not None:
        table = np.zeros((len(tiling_elements.keys()), 7))
    else:
        table = np.zeros((len(tiling_elements.keys()), 6))
    best_tilingelement = None
    best_snr = 0.0
    for i, (tilingelement, layer) in enumerate(tiling_elements.iteritems()):
        te_supp = tilingelement.support
        u_I, v_I = approximate_solve_mp_fixed_support(te_supp, tiling.A,
                                                      tiling.y)
        table[i, 0] = tilingelement.identifier
        table[i, 1] = len(te_supp)
        table[i, 2] = np.min(np.abs(u_I[te_supp]))  # signal strength
        table[i, 3] = np.max(np.abs(v_I))  # noise level/noise strength
        table[i, 4] = table[i, 2] / table[i, 3] # SNR
        table[i, 5] = layer
        if target_support is not None:
            # Set table[i,6] to symmetric support difference
            table[i, 6] = len(np.setdiff1d(te_supp, target_support)) + \
                len(np.setdiff1d(target_support, te_supp))
        if table[i, 4] > best_snr:
            best_snr = table[i, 4]
            best_tilingelement = tilingelement
    # Sort table
    ranking = np.argsort(table[:, 4])
    table = table[ranking, :]
    if show_table:
        header = ["Identifier", "# Active", "c", "d", "SNR", "Layer",
                  "Symmetric Diff"]
        print tabulate(table, headers=header)
    elapsed_time = timer() - starttime
    return table, best_tilingelement, elapsed_time
