# coding: utf8
import json
import os

import numpy as np

from problem_factory.synthetic_random_data import \
    create_specific_problem_data_from_problem
from support_selection.snr_based import highest_support_constrained_snr
from tiling import wrapper_create_tiling


def run_single(problem):
    """ Create tiling of a problem of type

        A * (u + v) = y + eps

    with randomly created data A, u, v and eps. The run characteristics (ie.
    noise levels, noise types, signal and noise strength and so forth) are
    given in the dictionary called 'problem'. Also, the
    dictionary stores other important characteristics of the run. Concretely,
    the dictonary must contain the following information:

    Dictionary-key | Description
    ---------------------------
    identifier | Subfolder identifier where the results shall be stored. Full
                 path will be "/results/<identifier>/".
    tiling_options | Options for the tiling object. See documentation of the
                     tiling class to see what options can be specified for
                     this class.
    beta_min | Minimum beta for which we want to find the supports along the
               Lasso path.
    beta_max | Maximum beta for which we want to find the supports along the
               Lasso path.
    upper_bound_tilingcreation | If this support length is reached in the
                                 tiling creation for a specific tiling element,
                                 we will not search for further childs of such
                                 a tiling element.

    For creation of random data (check the respective files to see what options
    for specific keys are available, and what specific options are used for).

    n_measurements | Number of measurements.
    n_features | Number of features.
    sparsity_level | Sparsity level of the correct u in A(u+v) = y + eps
    smallest_signal | Minimal signal strength min(|u_i|, i in supp(u))
    largest_signal | Maximal signal strength max(|u_i|, i in supp(u))
    noise_type_signal | Type of noise that is applied to the signal (ie. type
                        of noise of v).
    noise_lev_signal | Noise level of the signal noise.
    noise_type_measurements | Type of noise that is applied to the measurements
                              y (ie. type of noise of eps).
    noise_lev_measurements | Noise level of the measurement noise.
    random_seed | Random seed for the data creation. If given and fixed, the
                  same random data is created.

    Method will save the results to a file called data.npz
    in the folder 'results_single/<identifier>/'.
    If the file already exists, the file will be overwritten. Therefore, be
    careful if running several times with equal identifier.

    Returns
    ----------
    tiling : object of class Tiling
        The tiling created in the run

    best_tilingelement : Highest ranked tiling element according to the
        specified support selection procedure.
    """
    resultdir = 'results_single/' + problem['identifier'] + '/'
    if not os.path.exists(resultdir):
        os.makedirs(resultdir)
    with open(resultdir + 'log.txt', "w") as f:
        json.dump(problem, f, sort_keys=True, indent=4)
        f.write("\n")

    np.random.seed(problem["random_seed"])
    # Arguments to run the tiling creation
    tiling_options = problem["tiling_options"]
    beta_min = problem["beta_min"]
    beta_max = problem["beta_max"]
    upper_bound_tilingcreation = problem["upper_bound_tilingcreation"]
    random_state = np.random.get_state()
    problem["random_state"] = random_state
    # Creating problem data
    A, y, u_real, v_real = create_specific_problem_data_from_problem(
        problem)
    target_support = np.where(u_real)[0]
    tiling = wrapper_create_tiling(A, y, beta_min,
                                   beta_max,
                                   upper_bound_tilingcreation,
                                   options=tiling_options)
    tab = tiling.tabularise_results(u_real_for_comparison=u_real)
    elapsed_time_svd = tiling.elapsed_time_svd
    elapsed_time_tiling = tiling.elapsed_time_tiling
    ranking, best_tilingelement, elapsed_time_selection = \
    highest_support_constrained_snr(tiling, show_table=True,
                                    target_support=target_support)
    # Postprocessing of results
    # Total elapsed time
    elapsed_time = elapsed_time_svd + elapsed_time_tiling + \
        elapsed_time_selection
    # Flag signalising whether or not a the tiling contains the 'real support'
    tiling_contains_real = (len(np.where(ranking[:, 5] == 0)[0]) > 0)
    # Flag signalising whether correct support is connected to the highest
    # ranked tiling element
    highest_ranked_is_real = (ranking[-1, 5] == 0)
    np.savez_compressed(resultdir + "data.npz",
                        tabularised_results=tab,
                        elapsed_time=elapsed_time,
                        symmetric_difference=ranking[-1, 5],
                        support=best_tilingelement.support,
                        tiling_contains_real=tiling_contains_real,
                        highest_ranked_is_real=highest_ranked_is_real)
    return tiling, best_tilingelement
