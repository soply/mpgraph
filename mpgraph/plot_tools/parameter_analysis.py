# coding: utf8
""" Plotting methods for analysing the distribution of optimal beta's in batch
    experiments. """

__author__ = "Timo Klock"
import json

import matplotlib.pyplot as plt
import numpy as np

from scipy.misc import comb

__color_rotation__ = ['b','g','r','c','m','y','k']
__marker_rotation__ = ['o', 'H', 's', '^', 'None', '+', '.', 'D', 'x']
__linestyle_rotation__ = ['-', '--', ':', '-.']

def parameter_vs_sparsity_level(basefolder, identifier, methods, title = None,
                              xlabel = None, ylabel = None, save_as = None,
                              leg_loc = 'lower right'):
    """ Plots difference between minimum and maximum beta of the selected
    tiling, provided that the tiling was selected correctly, against the
    sparsity level of the experiments.

    Parameters
    -------------
    basefolder : python string
        Basefolder of files.

    identifier : python string
        Identifier inside basefolder.

    methods : python list of strings
        Python list of strings with method to which the solution has been
        calculated. E.g. ['mp_LARS', 'mp_LASSO'] or one of them.

    title, optional : python string
        Optinal title of the plot.

    xlabel, optional : python string
        Optional xlabel of the plot.

    ylabel, optional : python string
        Optinal ylabel of the plot.

    save_as, optional : python string
        If given, saves the figure to the file provided under 'save_as'.

    leg_loc, optional : python string
        Location of legend, using matplotlib keywords.
    """
    folder_names = {}
    for method in methods:
        folder_names[method] = basefolder + "/" + method + "_" + identifier + "/"
    # Load problem data
    with open(folder_names[methods[0]] + 'log.txt') as data_file:
        problem = json.load(data_file)
    num_experiments = problem['num_tests']
    sparsity_levels = problem['sparsity_level']
    beta_min = np.zeros((len(sparsity_levels), len(methods)))
    beta_max = np.zeros((len(sparsity_levels), len(methods)))
    beta_difference = np.zeros((len(sparsity_levels), len(methods)))
    fig = plt.figure(figsize = (16,9))
    for i, method in enumerate(methods):
        for j, sparsity_level in enumerate(sparsity_levels):
            meta_results = np.load(folder_names[method] + str(j) +\
                                      "/meta.npz")
            highest_ranked_is_real = meta_results["highest_ranked_is_real"]
            successful_experiments = np.where(highest_ranked_is_real)[0]
            for k in successful_experiments:
                datafile = np.load(folder_names[method] + str(j) +\
                                          "/" + str(k) + "_data.npz")
                correct_support_index = np.where(datafile["tabularised_results"][:,6] == 0)[0][0]
                beta_difference[j, i] += \
                    datafile["tabularised_results"][correct_support_index,4] - \
                    datafile["tabularised_results"][correct_support_index,2]
                plt.scatter(sparsity_level, \
                    datafile["tabularised_results"][correct_support_index,4] - \
                    datafile["tabularised_results"][correct_support_index,2])
            if len(successful_experiments) > 0:
                beta_difference[j, i] = beta_difference[j, i]/float(len(successful_experiments))
    plt.semilogy(sparsity_levels, beta_difference, linewidth = 3.0)
    plt.xlim([-1, np.max(sparsity_levels) + 1])
    plt.legend(methods, loc = leg_loc, ncol = 2)
    if xlabel is None:
        plt.xlabel(r'Support size')
    else:
        plt.xlabel(xlabel)
    if ylabel is None:
        plt.ylabel(r'Difference beta_min to beta_max')
    else:
        plt.ylabel(ylabel)
    if title is None:
        plt.title(r'Difference of optimal beta vs support size')
    else:
        plt.title(title)
    if save_as is not None:
        fig.savefig(save_as)
    plt.show()

def parameter_vs_fixed_sparsity_level(basefolder, identifier, methods,
                                      sparsity_level, title = None,
                                      xlabel = None, ylabel = None, save_as = None,
                                      leg_loc = 'lower right',
                                      yaxis_scaling = 'semilog'):
    """ Plots difference between minimum and maximum beta of the selected
    tiling, provided that the tiling was selected correctly, against the
    sparsity level of the experiments.

    Parameters
    -------------
    basefolder : python string
        Basefolder of files.

    identifier : python string
        Identifier inside basefolder.

    methods : python list of strings
        Python list of strings with method to which the solution has been
        calculated. E.g. ['mp_LARS', 'mp_LASSO'] or one of them.

    sparsity_level : python int
        The sparsity level for which the different runs shall be plotted.

    title, optional : python string
        Optinal title of the plot.

    xlabel, optional : python string
        Optional xlabel of the plot.

    ylabel, optional : python string
        Optinal ylabel of the plot.

    save_as, optional : python string
        If given, saves the figure to the file provided under 'save_as'.

    leg_loc, optional : python string
        Location of legend, using matplotlib keywords.

    yaxis_scaling, optional : python string
        Can be 'semilog' or 'normal'.

    """
    folder_names = {}
    for method in methods:
        folder_names[method] = basefolder + "/" + method + "_" + identifier + "/"
    # Load problem data
    with open(folder_names[methods[0]] + 'log.txt') as data_file:
        problem = json.load(data_file)
    num_experiments = problem['num_tests']
    sparsity_levels = problem['sparsity_level']
    beta_min = np.zeros((num_experiments, len(methods)))
    beta_max = np.zeros((num_experiments, len(methods)))
    fig = plt.figure(figsize = (16,9))
    for i, method in enumerate(methods):
        j = np.where(np.array(sparsity_levels) == sparsity_level)[0][0]

        meta_results = np.load(folder_names[method] + str(j) +\
                                      "/meta.npz")
        highest_ranked_is_real = meta_results["highest_ranked_is_real"]
        successful_experiments = np.where(highest_ranked_is_real)[0]
        ctr = 0
        for k in successful_experiments:
            datafile = np.load(folder_names[method] + str(j) +\
                                    "/" + str(k) + "_data.npz")
            correct_support_index = np.where(datafile["tabularised_results"][:,6] == 0)[0][0]
            beta_min[ctr, i] = datafile["tabularised_results"][correct_support_index,2]
            beta_max[ctr, i] = datafile["tabularised_results"][correct_support_index,4]
            plt.scatter(ctr, datafile["tabularised_results"][correct_support_index,4],
                        c = __color_rotation__[i])
            plt.scatter(ctr, datafile["tabularised_results"][correct_support_index,2],
                        c = __color_rotation__[i])
            ctr += 1
        if yaxis_scaling == 'semilog':
            plt.semilogy(np.arange(ctr), beta_min[0:ctr,i], __color_rotation__[i])
            plt.semilogy(np.arange(ctr), beta_max[0:ctr,i], __color_rotation__[i])
        else:
            plt.plot(np.arange(ctr), beta_min[0:ctr,i], __color_rotation__[i])
            plt.plot(np.arange(ctr), beta_max[0:ctr,i], __color_rotation__[i])
    plt.legend(methods, loc = leg_loc, ncol = 2)
    if xlabel is None:
        plt.xlabel(r'No. Experiment')
    else:
        plt.xlabel(xlabel)
    if ylabel is None:
        plt.ylabel(r'$\beta_{min}$ and $\beta_{max}$')
    else:
        plt.ylabel(ylabel)
    if title is None:
        plt.title(r'$\beta_{min}$ and $\beta_{max}$ vs Experiment')
    else:
        plt.title(title)
    if save_as is not None:
        fig.savefig(save_as)
    plt.show()
