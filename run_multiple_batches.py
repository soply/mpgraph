# coding: utf8
""" Callable run script to perform numerous simulations with multiple
    constellations. """

__author__ = "Timo Klock"

import getopt
import os
import sys

from mpgraph.run_batch import print_meta_results
from mpgraph.run_multiple_batches import run_numerous_multiple_constellations


def main(argv, problem):
    """ Method to run a multiple batches of experiments. Problem characteristics,
    run characteristics and tiling creation options are specified below. Can be
    used from command line and as a method to call.

    Parameters
    --------------
    argv : python list with 4 elements and arguments to run simulation.
        Example: argv = ['t', 'run', 'i', 'test123']

    problem : python dictionary that contains the run characteristics.
        See problem_factory/synthetic_random_data docs for details on the run
        characteristics.
    """
    identifier = ''
    task = ''
    helpstr = ("===============================================================\n"
               "Run file by typing 'python run_multiple_batchs.py -t <task>"
               " -i <identifier>'.\n"
               "<task> can be 'run' to simula new batches or 'show' to show\n"
               "results of all runs belonging to a previous batch. \n"
               "<identifier> is an arbitrary folder name.\n"
               "The run characteristics are specified inside"
               " 'run_multiple_batches.py' file.\n"
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
    problem.update({'identifier' : identifier})
    if task == 'run':
        print ("Running multiple batch simulation. Results will be stored in"
               " subfolders of '{0}'.".format('results_multiple_batches/' +
                                              identifier + '/'))
        run_numerous_multiple_constellations(problem)
    elif task == 'show':
        ctr = 0
        resultsdir = 'results_multiple_batches/' + "mp_" + \
                problem['tiling_options']['mode'] + "_" + identifier + '/'
        print "\n\n\n\n\n\n\n"
        while os.path.exists(resultsdir + str(ctr) + "/0_data.npz"):
            print_meta_results(resultsdir + str(ctr) + "/")
            ctr += 1
        else:
            print ("\n\nFound {0} runs for identifier '{1}' and "
                   "basedirectory '{2}'.".format(str(ctr), identifier, resultsdir))
    else:
        print "Please add identifer and/or task. Run file as follows:\n"
        print helpstr
        sys.exit(2)

if __name__ == "__main__":
    tiling_options = {
        'verbose': 2,
        'mode': 'LASSO',
        'print_summary': False
    }
    problem = {
        'tiling_options': tiling_options, # Options
        'beta_min': 1e-6, # Lower beta bound
        'beta_max': 100.0, # Upper beta bound
        'num_tests': 100, # Repititions per fixed experiment
        'n_measurements': 70, # = m
        'n_features': 70, # = n
        'sparsity_level': [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 20, 25, 30, 35], # Considered support sizes
        'upper_bound_tilingcreation': [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 20, 25, 30, 35], # Sparsity oracle
        'smallest_signal': 1.5, # Lower bound for signal entries. One entry with smallest signal is ensured!
        'largest_signal': 10.0, # Upper bound for signal entries.
        'noise_type_signal': 'uniform_ensured_max', # Uniform sampling of entries of v + maximum will be taken.
        'noise_lev_signal': 0.2, # Bound for absolute of entires of v.
        'noise_type_measurements': 'gaussian', # Does not matter since we have no measurement noise
        'noise_lev_measurements': 0.0, # No measurement noise
        'random_seed': 78,
        'verbosity' : False,
        'sampling_matrix_type' : 'gaussian'
    }
    main(sys.argv[1:], problem)
