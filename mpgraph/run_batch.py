# coding: utf8
""" Methods to run and analyse repitions of experiments with synthetic random
    data. """

__author__ = "Timo Klock"

import json
import os

import numpy as np
from tabulate import tabulate

from problem_factory.pertubation_problem import \
    create_specific_problem_data_from_problem as create_data_pertubation
from problem_factory.unmixing_problem import \
    create_specific_problem_data_from_problem as create_data_unmixing
from support_selection.layer_based import \
    largest_support_occuring_in_each_layer
from support_selection.snr_based import highest_support_constrained_snr
from tiling import wrapper_create_tiling
from mp_utils import approximate_solve_mp_fixed_support

__available_problem_types__ = ['unmixing', 'pertubation']


def run_numerous_one_constellation(problem, results_prefix = None):
    """ Create numerous tilings of problems of type

        1) A * (u + v) = y + eps
        2) (A + E) u = y + eps

    with randomly created data. The run characteristics (ie.
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
    sparsity_oracle_in_ranking (optional) | If True we perform the ranking by using
                                            the correct support size as a sparsity
                                            oracle.

    For creation of random data (check the respective files to see what options
    for specific keys are available, and what specific options are used for).

    n_measurements | Number of measurements.
    n_features | Number of features.
    sparsity_level | Sparsity level of the correct u in A(u+v) = y + eps
    smallest_signal | Minimal signal strength min(|u_i|, i in supp(u))
    largest_signal | Maximal signal strength max(|u_i|, i in supp(u))
    noise_type_measurements | Type of noise that is applied to the measurements
                              y (ie. type of noise of eps).
    noise_lev_measurements | Noise level of the measurement noise.
    random_seed | Random seed for the data creation. If given and fixed, the
                  same random data is created.
    sampling_matrix_type | Type of sampling matrix. See random_matrices.py in
                           problem_factory folder to see available matrices.
    problem_type | The type of problem to solve. Problems of type 1) are called
                   'unmixing', problems of type 2) are called 'pertubation'.

    Moreover, dependent on the problem type, the following properties need to be
    specified as well.

    For problems of type 1):
    noise_type_signal | Type of noise that is applied to the signal (ie. type
                        of noise of v).
    noise_lev_signal | Noise level of the signal noise.

    For problems of type 2):
    pertubation_matrix_type | Type of pertubation matrix that is added to
                              A. Can take same values as the sampling matrix.
    pertubation_matrix_level | Scaling factor between pertubation matrix
                               and sampling matrix, ie. ||E||_2/||A||_2.

    Method will save the results of each single run to a file called i_data.npz
    in the folder 'results_batch/<identifier>/', or if a 'results_prefix' is
    given, it will be stored in '<results_prefix>/<identifier>/'.
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
    problem_type = problem["problem_type"]
    if problem.get('sparsity_oracle_in_ranking', False):
        sparsity_oracle_ranking = problem['sparsity_level']
    else:
        sparsity_oracle_ranking = None
    for i in range(problem['num_tests']):
        print "\nRun example {0}/{1}".format(i + 1, problem['num_tests'])
        random_state = np.random.get_state()
        problem["random_state"] = random_state
        # Creating problem data
        if problem_type == "unmixing":
            A, y, u_real, v_real = create_data_unmixing(problem)
            signal_to_signal_noise_ratio = np.linalg.norm(A.dot(u_real))/ \
                                                np.linalg.norm(A.dot(v_real))
        elif problem_type == "pertubation":
            A, y, u_real, E = create_data_pertubation(problem)
            signal_to_signal_noise_ratio = np.linalg.norm(A.dot(u_real))/ \
                                                np.linalg.norm(E.dot(u_real))
        else:
            raise RuntimeError("Problem type {0} not recognized. Available {1}".format(
                problem_type, __available_problem_types__))
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
                                        target_support=target_support,
                                        sparsity_oracle=sparsity_oracle_ranking)
            # Get number of supports per support size
            n_supports_per_size = np.zeros(upper_bound_tilingcreation)
            for j in range(upper_bound_tilingcreation):
                n_supports_per_size[j] = len(np.where(tab[:,5] == j)[0])
            # Postprocessing of results
            # Total elapsed time
            elapsed_time = elapsed_time_svd + elapsed_time_tiling + \
                elapsed_time_selection
            # Flag signalising whether or not a the tiling contains the 'real support'
            tiling_contains_real = (len(np.where(ranking[:, 6] == 0)[0]) > 0)
            # Flag signalising whether correct support is connected to the highest
            # ranked tiling element
            highest_ranked_is_real = (ranking[-1, 6] == 0)
            # Compute a prediction error
            u_I, v_I = approximate_solve_mp_fixed_support(
                                            best_tilingelement.support, A, y)
            prediction_error = np.linalg.norm(
                                    A[:,best_tilingelement.support].dot(
                                        u_I[best_tilingelement.support])-
                                    A[:, target_support].dot(
                                        u_real[target_support])) ** 2/ \
                                    (np.linalg.norm(A[:,target_support].dot(
                                        u_real[target_support])) ** 2)
            # Beta boundaries for highest_ranked tile and real_tile (if
            # attainable by multi-penalty regularization)
            beta_boundary_best_ranked = (best_tilingelement.beta_min,
                                         best_tilingelement.beta_max)
            if tiling_contains_real:
                beta_boundary_real = []
                for te_id in ranking[np.where(ranking[:, 6] == 0)[0],0]:
                    real_tile = tiling.get_tiling_element(te_id)
                    beta_boundary_real.append((real_tile.beta_min,
                                               real_tile.beta_max))
            else:
                beta_boundary_real = [(-1.0, -1.0)]
            np.savez_compressed(resultdir + str(i) + "_data.npz",
                                tabularised_results=tab,
                                elapsed_time=elapsed_time,
                                symmetric_difference=ranking[-1, 6],
                                symmetric_difference_best=np.min(tab[:, 6]),
                                symmetric_difference_best_fixed_size=np.min(
                                    tab[tab[:,5] == problem['sparsity_level'], 6]),
                                support=best_tilingelement.support,
                                tiling_contains_real=tiling_contains_real,
                                highest_ranked_is_real=highest_ranked_is_real,
                                n_supports_per_size=n_supports_per_size,
                                prediction_error=prediction_error,
                                betabound_best_ranked=beta_boundary_best_ranked,
                                betabound_real=beta_boundary_real,
                                ssnr=signal_to_signal_noise_ratio)
        else:
            # FIXME: Only for repairing symmetric differences
            datafile = np.load(resultdir + str(i) + "_data.npz")
            tab=datafile['tabularised_results']
            elapsed_time=datafile['elapsed_time']
            symmetric_difference=datafile['symmetric_difference']
            symmetric_difference_best=datafile['symmetric_difference_best']
            support=datafile['support']
            tiling_contains_real=datafile['tiling_contains_real']
            highest_ranked_is_real=datafile['highest_ranked_is_real']
            n_supports_per_size=datafile['n_supports_per_size']
            prediction_error=datafile['prediction_error']
            betabound_best_ranked=datafile['betabound_best_ranked']
            betabound_real=datafile['betabound_real']
            ssnr=datafile['ssnr']
            symmetric_difference_best_fixed_size=np.min(tab[tab[:,5] == \
                                                problem['sparsity_level'], 6])
            np.savez_compressed(resultdir + str(i) + "_data.npz",
                                tabularised_results=tab,
                                elapsed_time=elapsed_time,
                                symmetric_difference=symmetric_difference,
                                symmetric_difference_best=np.min(tab[:, 6]),
                                symmetric_difference_best_fixed_size=np.min(
                                    tab[tab[:,5] == problem['sparsity_level'], 6]),
                                support=support,
                                tiling_contains_real=tiling_contains_real,
                                highest_ranked_is_real=highest_ranked_is_real,
                                n_supports_per_size=n_supports_per_size,
                                prediction_error=prediction_error,
                                betabound_best_ranked=betabound_best_ranked,
                                betabound_real=betabound_real,
                                ssnr=signal_to_signal_noise_ratio)
    create_meta_results(resultdir)
    print_meta_results(resultdir)


def create_meta_results(folder):
    """ Method analyses the results of a batch of runs for a single
    constellation. The files that correspond to these runs should be
    contained in the given folder and be named as

        folder + <num_run> + _data.npz.

    where num_run runs from 0 to the number of runs. The data files should
    contain the information as saved for example in the
    'run_numerous_one_constellation' method. The meta results consist of

    correct_support_selection : Whether or not the chosen support by the support
                                selection method is correct.
    symmetric_difference : Symmetric difference.
    tiling_contains_real : Flag that is True if the tiling contains the
                           'real support', False else.
    highest_ranked_is_real : Flag that is True if the highest ranked support is
                             the 'real support', False else.
    elapsed_time : Totally elapsed time.
    n_supports_per_size : Matrix with number of different supports per support
                        size per experiment.

    They are stored in the given folder and named as meta.txt.

    Parameters
    ------------
    folder : string
        Foldername in which files <num_run> + _data.npz are stored.

    Remarks
    ------------
    -Format of meta results is a dictionary with the above mentioned keys.
    """
    correct_support_selection = []
    symmetric_difference = []
    symmetric_difference_best = []
    symmetric_difference_best_fixed_size = []
    tiling_contains_real = []
    highest_ranked_is_real = []
    elapsed_time = []
    prediction_error = []
    ssnr = []
    i = 0
    while os.path.exists(folder + str(i) + "_data.npz"):
        datafile = np.load(folder + str(i) + "_data.npz")
        n_supports_per_size_iter = datafile['n_supports_per_size']
        if i == 0:
            n_supports_per_size = n_supports_per_size_iter
        else:
            n_supports_per_size = np.vstack((n_supports_per_size,
                                            n_supports_per_size_iter))
        correct_support_selection.append((datafile['symmetric_difference'] == 0))
        symmetric_difference.append(datafile['symmetric_difference'])
        symmetric_difference_best.append(datafile['symmetric_difference_best'])
        symmetric_difference_best_fixed_size.append(datafile['symmetric_difference_best_fixed_size'])
        tiling_contains_real.append(datafile['tiling_contains_real'])
        highest_ranked_is_real.append(datafile['highest_ranked_is_real'])
        elapsed_time.append(datafile['elapsed_time'])
        prediction_error.append(datafile['prediction_error'])
        ssnr.append(datafile['ssnr'])
        i = i + 1
    np.savez_compressed(folder + "meta",
                        correct_support_selection=np.array(correct_support_selection),
                        symmetric_difference=np.array(symmetric_difference),
                        symmetric_difference_best=symmetric_difference_best,
                        symmetric_difference_best_fixed_size=symmetric_difference_best_fixed_size,
                        tiling_contains_real=np.array(tiling_contains_real),
                        highest_ranked_is_real=np.array(highest_ranked_is_real),
                        elapsed_time=elapsed_time,
                        n_supports_per_size=n_supports_per_size,
                        prediction_error=prediction_error,
                        ssnr=ssnr)

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
    4) Statistics about number of different supports per support sizes.

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
    print "\n4) Number of tiling statistics:"
    n_supports_per_size = meta_results["n_supports_per_size"]
    summary_table = np.vstack((np.arange(n_supports_per_size.shape[1]),
                              np.mean(n_supports_per_size, 0)))
    summary_table = np.vstack((summary_table, np.min(n_supports_per_size, 0)))
    summary_table = np.vstack((summary_table, np.max(n_supports_per_size, 0)))
    # Transpose such that data is in the columns
    summary_table = summary_table.T
    print "Average number of support per support size:"
    print tabulate(summary_table, headers=['Size', 'Average #supports', 'Min.', 'Max'])
