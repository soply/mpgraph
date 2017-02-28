# coding: utf8
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
        'mode': 'LARS',
        'print_summary' : False
    }
    problem = {
        'tiling_options': tiling_options,
        'num_tests': 5,
        'beta_min': 1e-06,
        'beta_max': 100,
        'upper_bound_tilingcreation': 5,
        'n_measurements': 20,
        'n_features': 30,
        'sparsity_level': 5,
        'smallest_signal': 1.5,
        'largest_signal': 2.0,
        'noise_type_signal': 'uniform_ensured_max',
        'noise_lev_signal': 0.2,
        'noise_type_measurements': 'gaussian',
        'noise_lev_measurements': 0.0,
        'random_seed': 1223445
    }
    main(sys.argv[1:], problem)
