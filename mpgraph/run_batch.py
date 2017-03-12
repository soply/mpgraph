# coding: utf8
""" Methods to run and analyse repitions of experiments with synthetic random
    data. """

__author__ = "Timo Klock"

import json
import os

import numpy as np

from problem_factory.unmixing_problem import \
    create_specific_problem_data_from_problem
from support_selection.snr_based import highest_support_constrained_snr
from support_selection.layer_based import largest_support_occuring_in_each_layer
from tiling import wrapper_create_tiling


def run_numerous_one_constellation(problem, results_prefix = None):
    """ Create numerous tilings of problems of type

        A * (u + v) = y + eps

    with randomly created data A, u, v and eps. The run characteristics (ie.
    noise levels, noise types, signal and noise strength and so forth) are
    given in the dictionary called 'problem'. Also, the
    dictionary stores other important characteristics of the run. Concretely,
    the dictonary must contain the following information:

    Dictionary-key | Description
    ---------------------------
    identifier | Subfolder identifier where the results shall be stored. Full
                 path will be "/results_batch/<identifier>/", or
                 "<results_prefix>/<identifier>" if results_prefix is given.
    num_tests | Number of runs that shall be performed for the given
                characteristics.
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
    sampling_matrix_type | Type of sampling matrix. See random_matrices.py in
                           problem_factory folder to see available matrices.

    Method will save the results of each single run to a file called i_data.npz
    in the folder 'results_batch/<identifier>/', or if a 'results_prefix' is given, it
    will be stored in '<results_prefix>/<identifier>/'.
    If the file already exists, the specific run will be skipped (this is
    useful if we want to stop a run in the middle and restart it). At the end
    of the run, meta results over all runs are created and stored to a file
    called meta.txt in the same folder. This can be used to analyse a specific
    batch of runs.
    """
    if results_prefix is not None:
        resultdir = results_prefix + "mp_" + problem['tiling_options']['mode'] + \
                                            "_" + problem['identifier'] + '/'
    else:
        resultdir = 'results_batch/' + "mp_" + problem['tiling_options']['mode'] + \
                                            "_" +problem['identifier'] + '/'
    if not os.path.exists(resultdir):
        os.makedirs(resultdir)
    with open(resultdir + 'log.txt', "w") as f:
        json.dump(problem, f, sort_keys=True, indent=4)
        f.write("\n")

    meta_results = np.zeros((9, problem['num_tests']))
    np.random.seed(problem["random_seed"])
    # Arguments to run the tiling creation
    tiling_options = problem["tiling_options"]
    beta_min = problem["beta_min"]
    beta_max = problem["beta_max"]
    upper_bound_tilingcreation = problem["upper_bound_tilingcreation"]
    for i in range(problem['num_tests']):
        print "\nRun example {0}/{1}".format(i + 1, problem['num_tests'])
        random_state = np.random.get_state()
        problem["random_state"] = random_state
        # Creating problem data
        A, y, u_real, v_real = create_specific_problem_data_from_problem(
            problem)
        target_support = np.where(u_real)[0]
        if not os.path.exists(resultdir + str(i) + "_data.npz"):
            tiling = wrapper_create_tiling(A, y, beta_min,
                                           beta_max,
                                           upper_bound_tilingcreation,
                                           options=tiling_options)
            tab = tiling.tabularise_results(u_real_for_comparison=u_real)
            elapsed_time_svd = tiling.elapsed_time_svd
            elapsed_time_tiling = tiling.elapsed_time_tiling
            ranking, best_tilingelement, elapsed_time_selection = \
                highest_support_constrained_snr(tiling, show_table=False,
                                                target_support=target_support)
            # Postprocessing of results
            # Total elapsed time
            elapsed_time = elapsed_time_svd + elapsed_time_tiling + \
                elapsed_time_selection
            # Flag signalising whether or not a the tiling contains the 'real support'
            tiling_contains_real = (len(np.where(ranking[:, 6] == 0)[0]) > 0)
            # Flag signalising whether correct support is connected to the highest
            # ranked tiling element
            highest_ranked_is_real = (ranking[-1, 6] == 0)
            np.savez_compressed(resultdir + str(i) + "_data.npz",
                                tabularised_results=tab,
                                elapsed_time=elapsed_time,
                                symmetric_difference=ranking[-1, 6],
                                support=best_tilingelement.support,
                                tiling_contains_real=tiling_contains_real,
                                highest_ranked_is_real=highest_ranked_is_real)
    create_meta_results(resultdir)
    print_meta_results(resultdir)


def create_meta_results(folder):
    """ Method analyses the results of a batch of runs for a single
    constellations. The files that correspond to these runs should be
    contained in the given folder and be named as

        folder + <num_run> + _data.npz.

    where num_run runs from 0 to the number of runs. The data files should
    contain the information as saved for example in the
    'run_numerous_one_constellation' method. The meta results consist of

    0 : Whether or not the chosen support by the support selection method is
        correct.
    1 : Symmetric difference.
    2 : Flag that is True if the tiling contains the 'real support', False else.
    3 : Flag that is True if the highest ranked support is the 'real support',
        False else.
    4 : Totally elapsed time.

    They are stored in the given folder and named as meta.txt.

    Parameters
    ------------
    folder : string
        Foldername in which files <num_run> + _data.npz are stored.

    Remarks
    ------------
    -Format of meta results is one run per column, 5 rows corresponding to the
     above mentioned characteristics.
    """
    correct_support_selection = []
    symmetric_difference = []
    tiling_contains_real = []
    highest_ranked_is_real = []
    elapsed_time = []
    i = 0
    while os.path.exists(folder + str(i) + "_data.npz"):
        datafile = np.load(folder + str(i) + "_data.npz")
        correct_support_selection.append((datafile['symmetric_difference'] == 0))
        symmetric_difference.append(datafile['symmetric_difference'])
        tiling_contains_real.append(datafile['tiling_contains_real'])
        highest_ranked_is_real.append(datafile['highest_ranked_is_real'])
        elapsed_time.append(datafile['elapsed_time'])
        i = i + 1
    np.savez_compressed(folder + "meta",
                        correct_support_selection=np.array(correct_support_selection),
                        symmetric_difference=np.array(symmetric_difference),
                        tiling_contains_real=np.array(tiling_contains_real),
                        highest_ranked_is_real=np.array(highest_ranked_is_real),
                        elapsed_time=elapsed_time)

def print_meta_results(folder):
    """ Method to print out the meta results to the terminal. The print-out
    shows:
    1) Percentages of successful cases, percentage in which the tiling contains
       the 'real support', and cases in which the highest ranked support is the
       'real support'.
    2) Statistics about the time for specific runs (mean, variance, min, max).
    3) Suspicious cases with the failure reason: either the support is not
       found in the tiling at all; or the highest ranked support due to the
       ranking method selects another support.

    Parameters
    ------------
    folder: string
        Foldername of where to the respective find 'meta.txt' file. Note that
        the full searched location is given by pwd+'<folder>/meta.txt'.
    """
    meta_results = np.load(folder + "/meta.npz")
    num_tests = meta_results["elapsed_time"].shape[0]
    print "================== META RESULTS ======================"
    print "1) Percentages:"
    print "Support at the end recovered: {0}".format(
            np.sum(meta_results["correct_support_selection"])/float(num_tests))
    print "Tiling contains real support: {0}".format(
            np.sum(meta_results["correct_support_selection"])/float(num_tests))
    print "Highest ranked support is real: {0}".format(
            np.sum(meta_results["correct_support_selection"])/float(num_tests))
    print "\n2) Timings:"
    print "Avg = {0}    \nVariance = {1}  \n0.95-range = {2}   \nMin = {3}   \nMax = {4}".format(
        np.mean(meta_results["elapsed_time"]),
        np.var(meta_results["elapsed_time"]),
        [np.min(np.percentile(meta_results["elapsed_time"], 0.95)),
            np.max(np.percentile(meta_results["elapsed_time"], 95))],
        np.min(meta_results["elapsed_time"]),
        np.max(meta_results["elapsed_time"]))
    print "\n3) Suspicious cases:"
    incorrect_supp = np.where(meta_results["correct_support_selection"] == 0)[0]
    tiling_does_not_contain_real =  np.where(
                                meta_results["tiling_contains_real"] == 0)[0]
    highest_ranked_wrong = np.where(meta_results["highest_ranked_is_real"] == 0)[0]
    print "Examples support not correct: {0}".format(incorrect_supp)
    print "Symmetric differences unequal to zero: {0}".format(
            zip(incorrect_supp, meta_results["symmetric_difference"]
                                            [incorrect_supp]))
    print "Examples tiling does not contain real support {0}".format(
                                                tiling_does_not_contain_real)
    print "Examples highest ranked support is incorrect {0}".format(
                                                        highest_ranked_wrong)
