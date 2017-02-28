# coding: utf8
""" Callable run script to perform batch experiments"""

__author__ = "Timo Klock"

import getopt
import sys

from mp_graph.run_batch import (print_meta_results,
                                run_numerous_one_constellation)


def main(argv, problem):
    """ Method to run a batch of experiments. Problem characteristics, run
    characteristics and tiling creation options are specified below. Can be used
    from command line and as a method to call.

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
    problem.update({'identifier' : identifier, 'method' : 'mult_penal'})
    if task == 'run':
        print "Running batch simulation. Results will be stored in folder {0}".format(
            identifier)
        run_numerous_one_constellation(problem)
    elif task == 'show':
        folder = 'results_batch/' + "mp_" + \
                problem['tiling_options']['mode'] + "_" + identifier + '/'
        try:
            print_meta_results(folder)
        except IOError:
            print ("Could not load specified file. Check folder  "
                    "'results_batch/{0}' for meta file please.'".format(folder))
        finally:
            sys.exit(2)
    else:
        print "Please add identifer and/or task. Run file as follows:\n"
        print helpstr
        sys.exit(2)

if __name__ == "__main__":
    tiling_options = {
        'verbose': 2,
        'mode': 'LASSO',
        'print_summary' : False
    }
    problem = {
        'tiling_options': tiling_options,
        'num_tests': 100,
        'beta_min': 1e-06,
        'beta_max': 100,
        'upper_bound_tilingcreation': 5,
        'num_tests': 100, # Repititions per fixed experiment
        'n_measurements': 100, # = m
        'n_features': 50, # = n
        'sparsity_level': 50, # Considered support sizes
        'smallest_signal': 1.5, # Lower bound for signal entries. One entry with smallest signal is ensured!
        'largest_signal': 10.0, # Upper bound for signal entries.
        'noise_type_signal': 'uniform_ensured_max', # Uniform sampling of entries of v + maximum will be taken.
        'noise_lev_signal': 0.2, # Bound for absolute of entires of v.
        'noise_type_measurements': 'gaussian', # Does not matter since we have no measurement noise
        'noise_lev_measurements': 0.0, # No measurement noise
        'random_seed': 123123,
        'verbosity' : False
    }
    main(sys.argv[1:], problem)
