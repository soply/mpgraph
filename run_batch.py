# coding: utf8
import getopt
import json
import os
import sys

import numpy as np

from problem_factory.synthetic_random_data import \
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
        resultdir = results_prefix + problem['identifier'] + '/'
    else:
        resultdir = 'results_batch/' + problem['identifier'] + '/'
    if not os.path.exists(resultdir):
        os.makedirs(resultdir)
    with open(resultdir + 'log.txt', "a+") as f:
        json.dump(problem, f, sort_keys=True, indent=4)

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
                highest_support_constrained_snr(tiling, show_table=True,
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
    meta_results = np.zeros((0, 5))
    meta_results_tmp = np.zeros(5)
    i = 0
    while os.path.exists(folder + str(i) + "_data.npz"):
        datafile = np.load(folder + str(i) + "_data.npz")
        meta_results_tmp[0] = (datafile['symmetric_difference'] == 0)
        meta_results_tmp[1] = datafile['symmetric_difference']
        meta_results_tmp[2] = datafile['tiling_contains_real']
        meta_results_tmp[3] = datafile['highest_ranked_is_real']
        meta_results_tmp[4] = datafile['elapsed_time']
        meta_results = np.vstack((meta_results, meta_results_tmp))
        i = i + 1
    np.savetxt(folder + "meta.txt", meta_results)

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
    meta_results = np.loadtxt(folder + "/meta.txt")
    num_tests = meta_results.shape[0]
    meta_summary = np.sum(meta_results, axis=0) / float(num_tests)
    print "================== META RESULTS ======================"
    print "1) Percentages:"
    print "Support at the end recovered: {0}".format(meta_summary[0])
    print "Tiling contains real support: {0}".format(meta_summary[2])
    print "Highest ranked support is real: {0}".format(meta_summary[3])
    print "\n2) Timings:"
    print "Avg = {0}    \nVariance = {1}  \n0.95-range = {2}   \nMin = {3}   \nMax = {4}".format(
        np.mean(meta_results[:, 4]),
        np.var(meta_results[:, 4]),
        [np.min(np.percentile(meta_results[:, 4], 0.95)),
            np.max(np.percentile(meta_results[:, 4], 95))],
        np.min(meta_results[:, 4]),
        np.max(meta_results[:, 4]))
    print "\n3) Suspicious cases:"
    incorrect_supp = np.where(meta_results[:, 0] == 0)[0]
    tiling_does_not_contain_real =  np.where(meta_results[:, 2] == 0)[0]
    highest_ranked_wrong = np.where(meta_results[:, 3] == 0)[0]
    print "Examples support not correct: {0}".format(incorrect_supp)
    print "Symmetric differences unequal to zero: {0}".format(
                        zip(incorrect_supp, meta_results[incorrect_supp, 1]))
    print "Examples tiling does not contain real support {0}".format(
                                                tiling_does_not_contain_real)
    print "Examples highest ranked support is incorrect {0}".format(
                                                        highest_ranked_wrong)

def main(argv):
    identifier = ''
    task = ''
    helpstr = ("===============================================================\n"
               "Run file by typing 'python run_batch.py -t <task> -i <identifier>'.\n"
               "<task> can be 'run' to simula a new batch or 'show' to show\n"
               "results of a previous run. \n"
               "<identifier> is an arbitrary folder name.\n"
               "The run characteristics are specified inside 'run_batch.py' file.\n"
               "===============================================================\n")
    try:
        opts, args = getopt.getopt(argv, "ht:i:p:", ["task= identifier="])
    except getopt.GetoptError:
        print helpstr
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print helpstr
            sys.exit()
        elif opt in ("-i", "--identifier"):
            identifier = arg
        elif opt in ("-t", "--task"):
            task = arg
    if identifier == '':
        print "Please add identifer and/or task. Run file as follows:\n"
        print helpstr
        sys.exit(2)
    if task == 'run':
        print "Running batch simulation. Results will be stored in folder {0}".format(
            identifier)

        tiling_options = {
            'verbose': 2,
            'mode': 'LASSO',
            'print_summary' : False
        }
        problem = {
            'identifier': identifier,
            'tiling_options': tiling_options,
            'num_tests': 100,
            'beta_min': 1e-06,
            'beta_max': 100,
            'upper_bound_tilingcreation': 24,
            'n_measurements': 250,
            'n_features': 800,
            'sparsity_level': 15,
            'smallest_signal': 1.5,
            'largest_signal': 2.0,
            'noise_type_signal': 'linf_bounded',
            'noise_lev_signal': 0.05,
            'noise_type_measurements': 'gaussian',
            'noise_lev_measurements': 0.0,
            'random_seed': 1223445
        }
        run_numerous_one_constellation(problem)
    elif task == 'show':
        try:
            print_meta_results('results_batch/' + identifier + '/')
        except IOError:
            print ("Could not load specified file. Check folder  "
                    "'results_batch/{0}/' for meta file please.'".format(identifier))
        finally:
            sys.exit(2)
    else:
        print "Please add identifer and/or task. Run file as follows:\n"
        print helpstr
        sys.exit(2)

if __name__ == "__main__":
    main(sys.argv[1:])
