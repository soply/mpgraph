# coding: utf8
import getopt
import os
import sys

import numpy as np
from tabulate import tabulate

# from mp_param_optim_io import (load_result, plot_combined_results,
#                                plot_parameter_path, plot_results_for_alpha,
#                                plot_snr, plot_support_tiling, print_results,
#                                save_result, update_results)
# from mpsr_utils import (analyse_solution, c_d, indices_I, infinity_norm,
#                         solve_mp, sum_indices_of_bool_entries,
#                         symmetric_support_difference)
from problem_factory.synthetic_random_data import create_specific_problem_data
from tiling import wrapper_create_tiling


def perform_test(M, N, L, c_min, signal_noise_level, num_tests, identifier,
                 noise_type_signal, noise_type_measurements,
                 measurement_noise_level, n_layers,
                 c_max=None, random_seed=123456):
    # if not os.path.exists('results/' + identifier + '/'):
    #     os.makedirs('results/' + identifier + '/')
    # with open('results/' + identifier + '/log.txt', "a+") as f:
    #     f.write(("Testrun: " + identifier + "\n M: {0} N: {1} L: {2} c: {3}" +
    #              "d: {4} num_tests: {5} random_seed: {6} noise_type: {7}\n\n").format(
    #         M, N, L, c_min, signal_noise_level, num_tests, random_seed, noise_type_signal))
    """ Describe meta results """
    meta_results = np.zeros((9, num_tests))
    # Options for MultiPenaltySupportRecoverer
    options = {}
    options["verbose"] = 2
    np.random.seed(random_seed)
    beta_min = 1e-6
    beta_max = 100.0
    for i in range(num_tests):
        print "\nRun example {0}/{1}".format(i+1, num_tests)
        random_state = np.random.get_state()
        A, y, u_real, v_real = create_specific_problem_data(
                M, N, L, c_min,
                largest_signal = c_max,
                noise_type_signal = noise_type_signal,
                noise_lev_signal = signal_noise_level,
                noise_type_measurements = noise_type_measurements,
                measurement_noise_level = measurement_noise_level,
                random_seed = random_seed,
                random_state = random_state)
        if not os.path.exists('results/' + identifier + '/' + str(i) +
                              "_path.gz"):
            layers, tiling = wrapper_create_tiling(A, y, u_real, beta_min,
                                                   beta_max, L,
                                                   options = options)
        #     # try:
        #     exitflag_path, support, result_path, support_tiling, elapsed_time = \
        #         mpsr.run(n_layers, mode=mode)
        #     symmetric_difference = symmetric_support_difference(support,
        #                                                         indices_I(u_real, mode="idx"))
        #     S = np.linalg.svd(A, compute_uv = 0)
        #
        #     # print "min SVD: {0}     max SVD: {1}    sigma0/sigma1: {2}".format(
        #     #     np.min(S)** 2, np.max(S) ** 2, measurement_noise_level**2/(signal_noise_level ** 2))
        #     # beta_opt = (measurement_noise_level**2/(signal_noise_level ** 2) - np.min(S) ** 2)/ \
        #     #         (1.0 - measurement_noise_level**2/(signal_noise_level ** 2) * 1.0/(np.max(S) ** 2))
        #     # print "beta_opt = {0}".format(beta_opt)
        #     # Gather additional information about conditions if verification was used
        #     alpha_sat = mpsr.verification_alpha_below_boundary
        #     irr_sat = mpsr.verification_irrepresentable_condition_satisfied
        #     alpha_step_worked = mpsr.verification_alpha_stepping_worked
        #     ver_asleep = mpsr.verification_assleep
        #     # except np.linalg.linalg.LinAlgError:
        #     #     exitflag_path = 10
        #     #     elapsed_time = 1000
        #     #     result_path = mpsr._cut_path_results()
        #     #     support = np.zeros([])
        #     #     alpha_sat = False
        #     #     irr_sat = False
        #     #     alpha_step_worked = False
        #     #     ver_asleep = False
        #     #     symmetric_difference = L
        #
        #     np.savetxt("results/" + identifier + "/" + str(i) + "_path.gz",
        #                result_path)
        #     np.savetxt("results/" + identifier + "/" + str(i) + "_meta.gz",
        #                np.array([exitflag_path, elapsed_time, symmetric_difference,
        #                          alpha_sat, irr_sat, alpha_step_worked, ver_asleep]))
        # else:
        #     result_path = np.loadtxt("results/" + identifier + "/" + str(i) +
        #                              "_path.gz")
        #     meta_info = np.loadtxt("results/" + identifier + "/" + str(i) +
        #                            "_meta.gz")
        #     exitflag_path = meta_info[0]
        #     elapsed_time = meta_info[1]
        #     symmetric_difference = meta_info[2]
        #     alpha_sat = meta_info[3]
        #     irr_sat = meta_info[4]
        #     alpha_step_worked = meta_info[5]
        #     ver_asleep = meta_info[6]
        # # if i in [22]:
        # #     result_grid, real_support, recovered_support = \
        # #         load_result("for_presentation_0_0/199")[0:3]
        # #     meshgrid_size_alpha = len(np.unique(result_grid[:,1]))
        # #     meshgrid_size_beta = len(np.unique(result_grid[:,2]))
        # #     print_results(result_grid)
        # #     plot_combined_results(result_grid,
        # #         exact_support = real_support,
        # #         recovered_support = recovered_support,
        # #         path_results = None,
        # #         meshgrid_sizes = (meshgrid_size_alpha, meshgrid_size_beta))
        # #     options["maximum_beta"] = 16.0
        # #     mpsr = MultiPenaltySupportRecoverer(A, y, u_real, v_real,
        # #         options = options)
        # #     exitflag_path, elapsed_time, result_path, support_tiling, support = \
        # #         mpsr.run(n_layers, mode = mode)
        # #     symmetric_difference = symmetric_support_difference(support,
        # #         indices_I(u_real, mode = "idx"))
        # #     plot_support_tiling(support_tiling)
        #
        # # Check if support was recovered exactly at the end
        # if len(result_path.shape) == 1:
        #     result_path = np.array((result_path,))
        # if exitflag_path != 1:
        #     meta_results[0, i] = 1
        # else:
        #     meta_results[0, i] = 0
        # # Check if the conditions (1) and (2) are satisfied
        # if alpha_sat and irr_sat:
        #     meta_results[1, i] = 1
        # else:
        #     meta_results[1, i] = 0
        # # Check if C1 is smaller then 1
        # if irr_sat:
        #     meta_results[2, i] = 1
        # else:
        #     meta_results[2, i] = 0
        # # Check if alpha is larger then lower boundary
        # if result_path[-1, 1] >= result_path[-1, 10]:
        #     meta_results[3, i] = 1
        # else:
        #     meta_results[3, i] = 0
        # # Check if alpha is smaller then upper boundary
        # if alpha_sat:
        #     meta_results[4, i] = 1
        # else:
        #     meta_results[4, i] = 0
        # # Store the exitflag
        # meta_results[5, i] = int(exitflag_path)
        # # Store elapsed time
        # meta_results[6, i] = elapsed_time
        # # Store relative error
        # meta_results[7, i] = result_path[-1, 13]
        # # Store symmetric difference
        # meta_results[8, i] = int(symmetric_difference)
        # # Store meta results
        # np.savetxt("results/" + identifier + "/" + "meta.gz",
        #            meta_results)


# def show_meta_results(identifier):
#     meta_results = np.loadtxt("results/" + identifier + "/" + "meta.gz")
#     num_tests = meta_results.shape[1]
#     # Calculate percentages if necessary
#     meta_summary = np.sum(meta_results[0:5, :], axis=1) / float(num_tests)
#     # C1 percentage
#     meta_summary[2] = len(np.where(meta_results[5, :] != 7)[0]) / float(num_tests)
#     # Average symmetric difference
#     if len(meta_results[5, :] != 0) > 0:
#         avg_symm_diff = np.mean(meta_results[8, meta_results[5, :] != 0])
#     else:
#         avg_symm_diff = 0.0
#     # Present percentages
#     print "================== META RESULTS ======================"
#     print "1) Percentages:"
#     print "Support at the end recovered: {0}".format(meta_summary[0])
#     print "Both conditions satisfied: {0}".format(meta_summary[1])
#     print "C1 < 1: {0}".format(meta_summary[2])
#     print "Alpha > LB: {0}".format(meta_summary[3])
#     print "Alpha < UB: {0}".format(meta_summary[4])
#     print "\n2) Timings:"
#     print "Avg = {0}    \nVariance = {1}  \n0.95-range = {2}   \nMin = {3}   \nMax = {4}".format(
#         np.mean(meta_results[6, :]),
#         np.var(meta_results[6, :]),
#         [np.min(np.percentile(meta_results[6, :], 0.95)),
#             np.max(np.percentile(meta_results[6, :], 95))],
#         np.min(meta_results[6, :]),
#         np.max(meta_results[6, :]))
#     print "\n3) Relative error:"
#     print "Avg = {0}    \nVariance = {1}  \n0.95-range = {2}   \nMin = {3}   \nMax = {4}".format(
#         np.mean(meta_results[7, :]),
#         np.var(meta_results[7, :]),
#         [np.min(np.percentile(meta_results[7, :], 0.95)),
#             np.max(np.percentile(meta_results[7, :], 95))],
#         np.min(meta_results[7, :]),
#         np.max(meta_results[7, :]))
#     print "For successful cases only:"
#     if len(meta_results[7, meta_results[5, :] == 0]) > 0:
#         print "Avg = {0}    \nVariance = {1}  \n0.95-range = {2}   \nMin = {3}   \nMax = {4}".format(
#             np.mean(meta_results[7, meta_results[5, :] == 0]),
#             np.var(meta_results[7, meta_results[5, :] == 0]),
#             [np.min(np.percentile(meta_results[7, meta_results[5, :] == 0], 0.95)),
#                 np.max(np.percentile(meta_results[7, meta_results[5, :] == 0], 95))],
#             np.min(meta_results[7, meta_results[5, :] == 0]),
#             np.max(meta_results[7, meta_results[5, :] == 0]))
#     else:
#         print "No successful cases :("
#
#     # Present single index cases
#     print "\n4) Suspicious cases:"
#     print "Examples support not correct: {0}".format(np.where(meta_results[0, :] == 0)[0])
#     print "Examples conditions not true: {0}".format(np.where(meta_results[1, :] == 0)[0])
#     print "Examples C1 > 1: {0}".format(np.where(meta_results[5, :] == 7)[0])
#     print "Examples alpha > UB: {0}".format(np.where(meta_results[4, :] == 0)[0])
#     print "Examples support not correct but conditions true: {0}".format(
#         [i for i in range(num_tests) if (meta_results[0, i] == 0 and meta_results[1, i] == 1)]
#     )
#     print "All exit flags apart from 0-exitflag (index, exitflag, symmetric_difference): #wrong={0}:    {1}".format(
#         len([(i, meta_results[5, i]) for i in range(num_tests) if meta_results[5, i] != 0]),
#         [(i, meta_results[5, i], meta_results[8, i])
#          for i in range(num_tests) if meta_results[5, i] != 0]
#     )
#     print "Average symmetric difference of wrong examples: ", avg_symm_diff


def main(argv):
    task = ''
    identifier = ''
    try:
        opts, args = getopt.getopt(argv, "ht:i:p:", ["task=", "identifier="])
    except getopt.GetoptError:
        print "================================================================="
        print 'run.py -t <task> -i <identifier>'
        print "help me"
        print "==========================================================="
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print "==========================================================="
            print 'help me'
            print "==========================================================="
            sys.exit()
        elif opt in ("-t", "--task"):
            task = arg
        elif opt in ("-i", "--identifier"):
            identifier = arg
    print 'Task is "', task
    print 'Identifier is "', identifier
    # elif task == 'plot_batch':
    #     show_meta_results(identifier)
    if task == 'run_single':
        M = 200
        N = 500
        sparsity = 10
        c_min = 1.5
        c_max = 4.5
        signal_noise_level = 0.3
        measurement_noise_level = 0.20
        random_seed = 123
        noise_type_signal = "linf_bounded"
        noise_type_measurements = "gaussian"
        np.random.seed(random_seed)
        random_state = np.random.get_state()
        A, y, u_real, v_real = create_specific_problem_data(
                M, N, sparsity, c_min,
                largest_signal = c_max,
                noise_type_signal = noise_type_signal,
                noise_lev_signal = signal_noise_level,
                noise_type_measurements = noise_type_measurements,
                noise_lev_measurements = measurement_noise_level,
                random_seed = random_seed,
                random_state = random_state)
        beta_min = 1e-6
        beta_max = 100.0
        n_layers = sparsity
        layers, tiling = wrapper_create_tiling(A, y, u_real, beta_min,
                                               beta_max, n_layers)
    elif task == 'run_batch':
        N = [(450, 500)]
        sparsity_levels = [10]
        c_min = 1.5
        c_max = 4.5
        noise_lev_signal = 0.3
        measurement_noise_level = 0.20
        random_seeds = [1, 234, 345, 456]
        noise_type_signal = "linf_bounded"
        noise_type_measurements = "gaussian"
        num_tests = 100
        for i, L in enumerate(sparsity_levels):
            # for j in range(len(N)):
            perform_test(N[i][0],
                         N[i][1],
                         L,
                         c_min,
                         noise_lev_signal,
                         num_tests,
                         identifier + '_' + str(i) + '_' + str(i),
                         noise_type_signal,
                         noise_type_measurements,
                         measurement_noise_level,
                         n_layers=L,
                         c_max=c_max,
                         random_seed=random_seeds[i])

if __name__ == "__main__":
    main(sys.argv[1:])
