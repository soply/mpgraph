# coding: utf8
import getopt
import json
import os
import sys

from run_batch import print_meta_results, run_numerous_one_constellation


def run_numerous_multiple_constellations(problem):
    """ This method is similar to 'run_batch.run_numerous_one_constellation'
    with the difference that the problem dictionary (see docs of
    'run_batch.run_numerous_one_constellation' for more details on the dict) can
    be lists instead of single number. This enables to run multiple batches of
    different run characteristics, ie. different sparsity levels, feature sizes
    or measurement sizes.
    Concretely, the dictioniary should contain the same keys as decribed in
    'run_batch.run_numerous_one_constellation'. Each entry should either be
    a single object/value, if the respective characteristic is the same for all
    runs; or it should contain a list of different characteristics that should
    be iterated through. It is important to note that the dictionary can not
    contain multiple lists of different sizes because we iterate simoultaneously
    through all lists of the dictionary (if it is a list). Therefore all lists
    in the dictionary have to be equally sized.
    The results of the respective runs are saved to
    'results_multiple_batches/<problem['identifier']>/<i>/'
    where <i> is the number of the respective run.

    Example
    ------------
    tiling_options = {
        'verbose': 1,
        'mode': 'LARS',
        'print_summary': False
    }
    problem = {
        'identifier': 'test1',
        'tiling_options': tiling_options,
        'num_tests': 20,
        'beta_min': 1e-6,
        'beta_max': 100.0,
        'upper_bound_tilingcreation': 9,
        'n_measurements': [350, 500, 750],
        'n_features': [1250, 1250, 1500],
        'sparsity_level': 8,
        'smallest_signal': 1.5,
        'largest_signal': 2.0,
        'noise_type_signal': 'uniform',
        'noise_lev_signal': 0.3,
        'noise_type_measurements': 'gaussian',
        'noise_lev_measurements': 0.0,
        'random_seed': 1
    }
    'run_numerous_multiple_constellations(problem)'

    runs three seperated batch
    runs, where all characteristics except 'n_measurements' and 'n_features'
    remain the same for all runs. The latter two charactertics will be
    (350, 1250), (500, 1250), (750,1500) for these three runs.

    Remarks
    ----------------
    Please consult the docs of 'run_batch.run_numerous_one_constellation' for
    more information on the problem dictionary.
    """
    parentdir = "results_multiple_batches/" + "mp_" + \
                problem['tiling_options']['mode'] + "_" + problem['identifier']
    if not os.path.exists(parentdir):
        os.makedirs(parentdir)
    # Write log file to parent folder with problem description
    with open(parentdir + '/log.txt', "w") as f:
        json.dump(problem, f, sort_keys=True, indent=4)
        f.write("\n")
    # Check if problem dictionary contains either a single object, or a list of
    # the same size for each dictionary entry.
    listsize = 1
    for key, val in problem.iteritems():
        if isinstance(val, list):
            if listsize == 1:
                # No list found in dict so far. Set dictsize
                listsize = len(val)
                print key, listsize
            elif len(val) != listsize:
                raise RuntimeError(("The 'problem' dictionary does not contain"
                                    " equally sized lists. Each entry of the dictionary"
                                    " should either be a single entry or a list of size that"
                                    " consides with all lists in the dictionary.\n"
                                    "Problematic (key,val) : ({0},{1})".format(key, val)))
    # Get's i-th element if list or object if not a list
    problemgetter = lambda x, i: x[i] if isinstance(x, list) else x
    for i in range(listsize):
        subproblem = {key: problemgetter(problem[key], i) for key in
                      problem.keys()}
        subproblem['identifier'] += "/{0}/".format(str(i))
        run_numerous_one_constellation(subproblem,
                                       results_prefix="results_multiple_batches/")


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
        resultsdir = 'results_multiple_batches/' + identifier + '/'
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
        'num_tests': 20,
        'beta_min': 1e-6,
        'beta_max': 100.0,
        'upper_bound_tilingcreation': 9,
        'n_measurements': [350, 500, 750],
        'n_features': [1250, 1250, 1500],
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
