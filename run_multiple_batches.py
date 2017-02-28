# coding: utf8
import getopt
import os
import sys

from mp_graph.run_multiple_batches import run_numerous_multiple_constellations
from mp_graph.run_batch import print_meta_results


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
        'verbose': 1,
        'mode': 'LARS',
        'print_summary': False
    }
    problem = {
        'tiling_options': tiling_options,
        'num_tests': 5,
        'beta_min': 1e-6,
        'beta_max': 100.0,
        'upper_bound_tilingcreation': 9,
        'n_measurements': [10, 20],
        'n_features': [10, 20],
        'sparsity_level': 8,
        'smallest_signal': 1.5,
        'largest_signal': 2.0,
        'noise_type_signal': 'uniform_ensured_max',
        'noise_lev_signal': 0.3,
        'noise_type_measurements': 'gaussian',
        'noise_lev_measurements': 0.0,
        'random_seed': 1
    }
    main(sys.argv[1:], problem)
