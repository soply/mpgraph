# coding: utf8
""" Callable run script to perform single experiments. """

__author__ = "Timo Klock"

import getopt
import sys

from mpgraph.run_single import run_single


def main(argv, problem):
    """ Method to run a single experiment. Problem characteristics, run
    characteristics and tiling creation options are specified below. Can be used
    from command line and as a method to call.

    Parameters
    --------------
    argv : python list with 8 elements and arguments to run simulation.
        Example: argv = ['i', 'test', 'v', 'true', 'p', 'graph-layered']

    problem : python dictionary that contains the run characteristics.
        See problem_factory/synthetic_random_data docs for details on the run
        characteristics.
    """
    identifier = ''
    verification = False
    plotting = 'no'
    helpstr = ("===============================================================\n"
               "Run file by typing 'python run_single.py -i <identifier> -v "
               "<verification> -p <plotting>.\n"
               "<identifier> is an arbitraray folder name.\n"
               "<verification> can be true of false and defines whether or not\n"
               "the resulting tiling shall be verified (default: false).\n"
               "<plotting> can be 'graph', 'graph-layered', 'tiling' or 'no' \n"
               "and defines the method of how we display the reconstructed tiling\n"
               "(default: 'no').\n"
               "Note that some visualisation methods are rather costly"
               "The run characteristics are specified inside 'run_single.py' file.\n"
               "===============================================================\n")
    try:
        opts, args = getopt.getopt(argv, "i:v:p:h", ["identifier=",
                                                     "verification=",
                                                     "plotting=",
                                                     "help"])
    except getopt.GetoptError:
        print helpstr
        sys.exit(2)

    for opt, arg in opts:
        if opt in ('-h', "--help"):
            print helpstr
            sys.exit()
        elif opt in ("-i", "--identifier"):
            identifier = arg
        elif opt in ("-v", "--verification"):
            if arg == 'true':
                verification = True
        elif opt in ("-p", "--plotting"):
            plotting = arg
    if identifier == '':
        print "Please add identifer and/or task. Run file as follows:\n"
        print helpstr
        sys.exit(2)
    print "Running single simulation. Results will be stored in folder {0}".format(
        identifier)
    problem.update({'identifier' : identifier})
    tiling, best_tilingelement = run_single(problem)
    if verification:
        tiling.verify_tiling()
    if plotting == 'graph':
        tiling.plot_tiling_graph(y_mode = 'alpha')
    elif plotting == "graph-layered":
        tiling.plot_tiling_graph(y_mode = 'layered')
    elif plotting == "tiling":
        tiling.plot_tiling(n_disc = 3)
    else:
        print "Plotting method {0} not recognized.".format(plotting)

if __name__ == "__main__":
    tiling_options = {
        'verbose': 2,
        'mode': 'LASSO',
        'print_summary' : True
    }
    problem = {
        'tiling_options': tiling_options, # Options
        'beta_min': 1e-6, # Lower beta bound
        'beta_max': 100.0, # Upper beta bound
        'n_measurements': 70, # = m
        'n_features': 70, # = n
        'sparsity_level': 20, # Considered support sizes
        'upper_bound_tilingcreation': 20, # Sparsity oracle
        'smallest_signal': 1.5, # Lower bound for signal entries. One entry with smallest signal is ensured!
        'largest_signal': 10.0, # Upper bound for signal entries.
        'noise_type_signal': 'uniform_ensured_max', # Uniform sampling of entries of v + maximum will be taken.
        'noise_lev_signal': 0.2, # Bound for absolute of entires of v.
        'noise_type_measurements': 'gaussian', # Does not matter since we have no measurement noise
        'noise_lev_measurements': 0.0, # No measurement noise
        'random_seed': 89,
        'sampling_matrix_type' : 'gaussian'
    }
    main(sys.argv[1:], problem)
