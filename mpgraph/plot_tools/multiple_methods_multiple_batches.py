# coding: utf8
""" Plotting methods for experiments with synthethic random data where we test
    multiple batches. """

__author__ = "Timo Klock"
import json

import matplotlib.pyplot as plt
import numpy as np

from scipy.misc import comb

__color_rotation__ = ['b','g','r','c','m','y','k']
__marker_rotation__ = ['o', 'H', 's', '^', 'None', '+', '.', 'D', 'x']
__linestyle_rotation__ = ['-', '--', ':', '-.']

def success_vs_sparsity_level(basefolder, identifier, methods, title = None,
                              xlabel = None, ylabel = None, save_as = None,
                              leg_loc = 'lower right'):
    """ Creates a plot success rate vs sparsity level plot. Sparsity level is
    on x axis and success rate on y. The method akes a list of methods as an
    input, so several methods can be compared.  The data is assumed to lie at

    '<basefolder>/<method>_<identifier>/<ctr_sparsity_level>/meta.npz"

    where <ctr_sparsity_level> is a counter from 0 to the number of different
    sparsity levels that can be found.

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
    sparsity_levels = problem['sparsity_level']
    success_rates = np.zeros((len(sparsity_levels), len(methods)))
    for i, method in enumerate(methods):
        for j, sparsity_level in enumerate(sparsity_levels):
            meta_results = np.load(folder_names[method] + str(j) +\
                                      "/meta.npz")
            num_tests = problem['num_tests']
            success_rates[j, i] = np.sum(meta_results["tiling_contains_real"])/ \
                                                float(num_tests)
    fig = plt.figure(figsize = (16,9))
    plt.plot(sparsity_levels, success_rates, linewidth = 3.0)
    plt.legend(methods, loc = leg_loc, ncol = 2)
    if xlabel is None:
        plt.xlabel(r'Support size')
    else:
        plt.xlabel(xlabel)
    if ylabel is None:
        plt.ylabel(r'Success rate in %')
    else:
        plt.ylabel(ylabel)
    if title is None:
        plt.title(r'Success rate vs support size')
    else:
        plt.title(title)
    plt.ylim([-0.05, 1.05])
    if save_as is not None:
        fig.savefig(save_as)
    plt.show()

def success_vs_signal_noise(basefolder, identifier, methods, title = None,
                            xlabel = None, ylabel = None, save_as = None,
                            leg_loc = 'lower right'):
    """ Creates a plot success rate vs signal to noise ratio (SNR) where the
    noise is applied directly to the signal (not on the measurements). SNR is
    on x axis and success rate on y. The method akes a list of methods as an
    input, so several methods can be compared.  The data is assumed to lie at

    '<basefolder>/<method>_<identifier>/<ctr_SNR>/meta.npz"

    where <ctr_SNR> is a counter from 0 to the number of different signal to
    noise ratios.

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
    signal_noise = np.array(problem['noise_lev_signal']).astype('float')
    smallest_signal_entry = problem['smallest_signal']
    signal_to_noise_ratios = smallest_signal_entry/signal_noise
    success_rates = np.zeros((len(signal_to_noise_ratios), len(methods)))
    for i, method in enumerate(methods):
        for j in range(len(signal_to_noise_ratios)):
            meta_results = np.load(folder_names[method] + str(j) +\
                                      "/meta.npz")
            num_tests = problem['num_tests']
            success_rates[j, i] = np.sum(meta_results["tiling_contains_real"])/ \
                                                float(num_tests)
    fig = plt.figure(figsize = (16,9))
    plt.semilogx(signal_to_noise_ratios, success_rates, linewidth = 3.0)
    plt.legend(methods, loc = leg_loc, ncol = 2)
    if xlabel is None:
        plt.xlabel(r'SNR (smallest signal entry/signal noise)')
    else:
        plt.xlabel(xlabel)
    if ylabel is None:
        plt.ylabel(r'Success rate in %')
    else:
        plt.ylabel(ylabel)
    if title is None:
        plt.title(r'Success rate vs SNR')
    else:
        plt.title(title)
    plt.ylim([-0.05, 1.05])
    if save_as is not None:
        fig.savefig(save_as)
    plt.show()

def success_vs_signal_gap(basefolder, identifier, methods, title = None,
                          xlabel = None, ylabel = None, save_as = None,
                          leg_loc = 'lower right'):
    """ Creates a plot success rate vs signal gap meaning the largest signal
    divided through the smallest signal appearning in the randoms signal. The
    signal gap  is on x axis and success rate on y.
    The method akes a list of methods as an input, so several methods can be
    compared.  The data is assumed to lie at

    '<basefolder>/<method>_<identifier>/<ctr_signal_gap>/meta.npz"

    where <ctr_signal_gap> is a counter from 0 to the number of different signal
    to noise ratios.

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
    largest_signal_entry = np.array(problem['largest_signal']).astype('float')
    smallest_signal_entry = problem['smallest_signal']
    signal_gaps = largest_signal_entry/smallest_signal_entry
    success_rates = np.zeros((len(signal_gaps), len(methods)))
    for i, method in enumerate(methods):
        for j in range(len(signal_gaps)):
            meta_results = np.load(folder_names[method] + str(j) +\
                                      "/meta.npz")
            num_tests = problem['num_tests']
            success_rates[j, i] = np.sum(meta_results["tiling_contains_real"])/ \
                                                float(num_tests)
    fig = plt.figure(figsize = (16,9))
    plt.semilogx(signal_gaps, success_rates, linewidth = 3.0)
    plt.legend(methods, loc = leg_loc, ncol = 2)
    if xlabel is None:
        plt.xlabel(r'Signal gap (Largest absoulute entry/Smallest absolute entry)')
    else:
        plt.xlabel(xlabel)
    if ylabel is None:
        plt.ylabel(r'Success rate in %')
    else:
        plt.ylabel(ylabel)
    if title is None:
        plt.title(r'Success rate vs Signal gap')
    else:
        plt.title(title)
    plt.ylim([-0.05, 1.05])
    if save_as is not None:
        fig.savefig(save_as)
    plt.show()

def success_plus_highest_ranked_vs_sparsity_level(basefolder, identifier, methods,
                          title = None, xlabel = None, ylabel = None,
                          save_as = None, leg_loc = 'lower right'):
    """ Creates a plot with the 'success rate' and 'highest_ranked_is_real'
    (that is true if the highest ranked support according to the snr_based
    ranking is the correct one) vs support size of the generating signal. The
    support size is on the x axis and success rate on y.
    The method takes a list of methods as an input, so several methods can be
    compared.  The data is assumed to lie at

    '<basefolder>/<method>_<identifier>/<ctr_support_size>/meta.npz"

    where <ctr_sparsity_level> is a counter from 0 to the number of different
    support sizes.

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
    sparsity_levels = problem['sparsity_level']
    success_rates = np.zeros((len(sparsity_levels), len(methods)))
    highest_ranked_real_rates = np.zeros((len(sparsity_levels), len(methods)))
    for i, method in enumerate(methods):
        for j in range(len(sparsity_levels)):
            meta_results = np.load(folder_names[method] + str(j) +\
                                      "/meta.npz")
            num_tests = problem['num_tests']
            success_rates[j, i] = np.sum(meta_results["tiling_contains_real"])/ \
                                                float(num_tests)
            highest_ranked_real_rates[j, i] = np.sum(
                        meta_results["highest_ranked_is_real"])/float(num_tests)
    fig = plt.figure(figsize = (16,9))
    plt.plot(sparsity_levels, success_rates, linewidth = 3.0)
    plt.plot(sparsity_levels, highest_ranked_real_rates, marker = "o",
                 linewidth = 3.0)
    legend = ["Success Tiling (" + method + ")" for method in methods]
    legend = legend+["Success Ranking (" + method + ")" for method in methods]
    plt.legend(legend, loc = leg_loc, ncol = 1)
    if xlabel is None:
        plt.xlabel(r'Support size')
    else:
        plt.xlabel(xlabel)
    if ylabel is None:
        plt.ylabel(r'Success rate, highest ranked equals in %')
    else:
        plt.ylabel(ylabel)
    if title is None:
        plt.title(r'Success rate and highest ranked equals vs support size')
    else:
        plt.title(title)
    plt.ylim([-0.05, 1.05])
    if save_as is not None:
        fig.savefig(save_as)
    plt.show()

def success_plus_highest_ranked_vs_signal_noise(basefolder, identifier, methods,
                          title = None, xlabel = None, ylabel = None,
                          save_as = None, leg_loc = 'lower right'):
    """ Creates a plot with the 'success rate' and 'highest_ranked_is_real'
    (that is true if the highest ranked support according to the snr_based
    ranking is the correct one) vs signal to noise ratio (SNR). The
    SNR is on the x axis and success rate on y.
    The method takes a list of methods as an input, so several methods can be
    compared.  The data is assumed to lie at

    '<basefolder>/<method>_<identifier>/<ctr_SNR>/meta.npz"

    where <ctr_SNR> is a counter from 0 to the number of different SNR's.

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
    signal_noise = np.array(problem['noise_lev_signal']).astype('float')
    smallest_signal_entry = problem['smallest_signal']
    signal_to_noise_ratios = smallest_signal_entry/signal_noise
    success_rates = np.zeros((len(signal_to_noise_ratios), len(methods)))
    highest_ranked_real_rates = np.zeros((len(signal_to_noise_ratios), len(methods)))
    for i, method in enumerate(methods):
        for j in range(len(signal_to_noise_ratios)):
            meta_results = np.load(folder_names[method] + str(j) +\
                                      "/meta.npz")
            num_tests = problem['num_tests']
            success_rates[j, i] = np.sum(meta_results["tiling_contains_real"])/ \
                                                float(num_tests)
            highest_ranked_real_rates[j, i] = np.sum(
                        meta_results["highest_ranked_is_real"])/float(num_tests)
    fig = plt.figure(figsize = (16,9))
    plt.semilogx(signal_to_noise_ratios, success_rates, linewidth = 3.0)
    plt.semilogx(signal_to_noise_ratios, highest_ranked_real_rates,
                 marker = "o", linewidth = 3.0)
    legend = ["Success Tiling (" + method + ")" for method in methods]
    legend = legend+["Success Ranking (" + method + ")" for method in methods]
    plt.legend(legend, loc = leg_loc, ncol = 1)
    if xlabel is None:
        plt.xlabel(r'SNR (smallest signal entry/signal noise)')
    else:
        plt.xlabel(xlabel)
    if ylabel is None:
        plt.ylabel(r'Success rate, highest ranked equals in %')
    else:
        plt.ylabel(ylabel)
    if title is None:
        plt.title(r'Success rate and highest ranked equals vs SNR')
    else:
        plt.title(title)
    plt.ylim([-0.05, 1.05])
    if save_as is not None:
        fig.savefig(save_as)
    plt.show()

def success_plus_highest_ranked_vs_signal_gap(basefolder, identifier, methods,
                          title = None, xlabel = None, ylabel = None,
                          save_as = None, leg_loc = 'lower right'):
    """ Creates a plot with the 'success rate' and 'highest_ranked_is_real'
    (that is true if the highest ranked support according to the snr_based
    ranking is the correct one) vs signal gap meaning the largest signal
    divided through the smallest signal appearning in the randoms signal. The
    signal gap  is on x axis and success rate on y.
    The method akes a list of methods as an input, so several methods can be
    compared.  The data is assumed to lie at

    '<basefolder>/<method>_<identifier>/<ctr_signal_gap>/meta.npz"

    where <ctr_signal_gap> is a counter from 0 to the number of different signal
    to noise ratios.

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
    largest_signal_entry = np.array(problem['largest_signal']).astype('float')
    smallest_signal_entry = problem['smallest_signal']
    signal_gaps = largest_signal_entry/smallest_signal_entry
    success_rates = np.zeros((len(signal_gaps), len(methods)))
    highest_ranked_real_rates = np.zeros((len(signal_gaps), len(methods)))
    for i, method in enumerate(methods):
        for j in range(len(signal_gaps)):
            meta_results = np.load(folder_names[method] + str(j) +\
                                      "/meta.npz")
            num_tests = problem['num_tests']
            success_rates[j, i] = np.sum(meta_results["tiling_contains_real"])/ \
                                                float(num_tests)
            highest_ranked_real_rates[j, i] = np.sum(
                        meta_results["highest_ranked_is_real"])/float(num_tests)
    fig = plt.figure(figsize = (16,9))
    plt.semilogx(signal_gaps, success_rates, linewidth = 3.0)
    plt.semilogx(signal_gaps, highest_ranked_real_rates, marker = "o",
                 linewidth = 3.0)
    legend = ["Success Tiling (" + method + ")" for method in methods]
    legend = legend+["Success Ranking (" + method + ")" for method in methods]
    plt.legend(legend, loc = leg_loc, ncol = 1)
    if xlabel is None:
        plt.xlabel(r'Signal gap (Largest absoulute entry/Smallest absolute entry)')
    else:
        plt.xlabel(xlabel)
    if ylabel is None:
        plt.ylabel(r'Success rate, highest ranked equals in %')
    else:
        plt.ylabel(ylabel)
    if title is None:
        plt.title(r'Success rate and highest ranked equals vs signal gap')
    else:
        plt.title(title)
    plt.ylim([-0.05, 1.05])
    if save_as is not None:
        fig.savefig(save_as)
    plt.show()

def success_plus_highest_ranked_vs_measurement_noise(basefolder, identifier, methods,
                          title = None, xlabel = None, ylabel = None,
                          save_as = None, leg_loc = 'lower right'):
    """ Creates a plot with the 'success rate' and 'highest_ranked_is_real'
    (that is true if the highest ranked support according to the snr_based
    ranking is the correct one) vs additive Gaussian measurement noise applied
    to y. The measurement noise level is on the y axis.
    The method akes a list of methods as an input, so several methods can be
    compared.  The data is assumed to lie at

    '<basefolder>/<method>_<identifier>/<ctr_measurement_noise>/meta.npz"

    where <ctr_measurement_noise> is a counter from 0 to the number of
    different measurement noises.

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
    smallest_signal_entry = problem['smallest_signal']
    measurement_noises = np.array(problem['noise_lev_measurements'])
    measurement_SNR = measurement_noises/smallest_signal_entry
    success_rates = np.zeros((len(measurement_SNR), len(methods)))
    highest_ranked_real_rates = np.zeros((len(measurement_SNR), len(methods)))
    for i, method in enumerate(methods):
        for j in range(len(measurement_SNR)):
            meta_results = np.load(folder_names[method] + str(j) +\
                                      "/meta.npz")
            num_tests = problem['num_tests']
            success_rates[j, i] = np.sum(meta_results["tiling_contains_real"])/ \
                                                float(num_tests)
            highest_ranked_real_rates[j, i] = np.sum(
                        meta_results["highest_ranked_is_real"])/float(num_tests)
    fig = plt.figure(figsize = (16,9))
    plt.semilogx(measurement_SNR, success_rates, linewidth = 3.0)
    plt.semilogx(measurement_SNR, highest_ranked_real_rates, marker = "o",
                 linewidth = 3.0)
    legend = ["Success Tiling (" + method + ")" for method in methods]
    legend = legend+["Success Ranking (" + method + ")" for method in methods]
    plt.legend(legend, loc = leg_loc, ncol = 1)
    if xlabel is None:
        plt.xlabel(r'$\sigma^2/c_{\rm{min}}$')
    else:
        plt.xlabel(xlabel)
    if ylabel is None:
        plt.ylabel(r'Success rate, highest ranked equals in %')
    else:
        plt.ylabel(ylabel)
    if title is None:
        plt.title(r'Success rate and highest ranked equals vs measurement noise')
    else:
        plt.title(title)
    plt.ylim([-0.05, 1.05])
    if save_as is not None:
        fig.savefig(save_as)
    plt.show()

def n_supports_at_largest_vs_sparsity_level(basefolder, identifier, methods, title = None,
                                            xlabel = None, ylabel = None, save_as = None,
                                            leg_loc = 'lower right',
                                            prescribed_legend = None, legend_cols = 2,
                                            legend_fontsize = 20.,
                                            plot_handles = None):
    """ Creates a plot with the number of found supports at the highest sparsity
    level parameterised over experiment with different vs sparsity levels plot.
    Sparsity level is on x axis and success rate on y. The method akes a list of
    methods as an input, so several methods can be compared.  The data is assumed
    to lie at

    '<basefolder>/<method>_<identifier>/<ctr_sparsity_level>/meta.npz"

    where <ctr_sparsity_level> is a counter from 0 to the number of different
    sparsity levels that can be found.

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

    prescribed_legend : list of python strings
        Prescribed legend instead of automatically created legend

    legend_cols : int
        Columns in the legend

    legend_fontsize : int
        Fontsize of the legend

    plot_handles : matplotlib figure and axis objects
        In case, an existing plot should be extended we can give the figure and
        axis handle.

        Example:
            fig, ax = plt.figure()
            plot_handles = (fig, ax)
    """
    folder_names = {}
    for method in methods:
        folder_names[method] = basefolder + "/" + method + "_" + identifier + "/"
    # Load problem data
    with open(folder_names[methods[0]] + 'log.txt') as data_file:
        problem = json.load(data_file)
    sparsity_levels = problem['sparsity_level']
    n_supports = np.zeros((len(sparsity_levels), len(methods)))
    for i, method in enumerate(methods):
        for j, sparsity_level in enumerate(sparsity_levels):
            meta_results = np.load(folder_names[method] + str(j) +\
                                      "/meta.npz")
            num_tests = problem['num_tests']
            n_supports_p_sparsity_level = np.sum(meta_results["n_supports_per_size"],
                                                 axis = 0)/float(num_tests)
            n_supports_last_sparsity_level = n_supports_p_sparsity_level[-1]
            n_supports[j, i] = n_supports_last_sparsity_level
    if plot_handles is None:
        fig = plt.figure(figsize = (16,9))
        ax = plt.gca()
        cycle_offset = 0
    else:
        fig, ax = plot_handles
        cycle_offset = len(ax.lines)
    for j in range(n_supports.shape[1]):
        ax.plot(sparsity_levels, n_supports[:,j], linewidth = 3.0,
                 c = __color_rotation__[(j+5+cycle_offset) % len(__color_rotation__)],
                 linestyle = __linestyle_rotation__[(j+5+cycle_offset) % len(__linestyle_rotation__)],
                 marker = __marker_rotation__[(j+5+cycle_offset) % len(__marker_rotation__)],
                 markersize = 15.0)
    if prescribed_legend is None:
        ax.legend(methods, loc = leg_loc, ncol = legend_cols,
                   fontsize = legend_fontsize)
    else:
        ax.legend(prescribed_legend, loc = leg_loc, ncol = legend_cols,
                   fontsize = legend_fontsize)
    if xlabel is None:
        ax.set_title(r'Support size')
    else:
        ax.set_xlabel(xlabel)
    if ylabel is None:
        ax.set_ylabel(r'Average number of tiles at last iteration')
    else:
        ax.set_ylabel(ylabel)
    if title is None:
        ax.set_title(r'Average number of tiles at last iteration vs support size')
    else:
        ax.set_title(title)
    if save_as is not None:
        fig.savefig(save_as)
    if plot_handles is None:
        ax.set_ylim([0.95, 1.1 * np.max(n_supports)])
        plt.show()

def n_supports_vs_tiling_size(basefolder, identifier, methods, varying_parameter,
                              title = None, xlabel = None, ylabel = None,
                              save_as = None,  leg_loc = 'lower right',
                              prescribed_legend = None, legend_cols = 2,
                              legend_fontsize = 20.0):
    """ Creates a plot with the average number of tiles per tiling size for each
    support size. Different curves can be parameterized by some parameter, eg.
    the noise_level or the signal gap. All involved runs should contain tilings
    that have been calculated up to the same support size. The files should be
    located in the folder

    '<basefolder>/<method>_<identifier>/<ctr_sparsity_level>/meta.npz"

    where <ctr_sparsity_level> is a counter from 0 to the number of different
    sparsity levels that can be found.

    Parameters
    -------------
    basefolder : python string
        Basefolder of files.

    identifier : python string
        Identifier inside basefolder.

    methods : python list of strings
        Python list of strings with method to which the solution has been
        calculated. E.g. ['mp_LARS', 'mp_LASSO'] or one of them.

    varying_parameter : pyton string
        Parameter that yield numerous runs, ie. noise_level_signal and so forth,...

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

    prescribed_legend : list of python strings
        Prescribed legend instead of automatically created legend

    legend_cols : int
        Columns in the legend

    legend_fontsize : int
        Fontsize of the legend
    """
    folder_names = {}
    for method in methods:
        folder_names[method] = basefolder + "/" + method + "_" + identifier + "/"
    # Load problem data
    with open(folder_names[methods[0]] + 'log.txt') as data_file:
        problem = json.load(data_file)
    varying_parameters = problem[varying_parameter]
    n_parameters = len(varying_parameters)
    if isinstance(problem['upper_bound_tilingcreation'], list):
        raise RuntimeError("Runs do not have same support size bound: {0}".format(
                n_supports))
    n_support_sizes = np.arange(problem['upper_bound_tilingcreation'])
    n_avg_supports = np.zeros((problem['upper_bound_tilingcreation'],
                               len(methods) * len(varying_parameters)))
    legend_entries = []
    for i, method in enumerate(methods):
        for j, parameter in enumerate(varying_parameters):
            meta_results = np.load(folder_names[method] + str(j) + "/meta.npz")
            num_tests = problem['num_tests']
            n_supports_p_sparsity_level = np.sum(meta_results["n_supports_per_size"],
                                                 axis = 0)/float(num_tests)
            n_avg_supports[:, i * n_parameters + j] = n_supports_p_sparsity_level
            legend_entries.append(method + ', para = ' + str(parameter))
    fig = plt.figure(figsize = (16,9))
    for j in range(n_avg_supports.shape[1]):
        plt.plot(n_support_sizes, n_avg_supports[:,j], linewidth = 3.0,
                c = __color_rotation__[(j+5) % len(__color_rotation__)],
                linestyle = __linestyle_rotation__[(j+5) % len(__linestyle_rotation__)],
                marker = __marker_rotation__[(j+5) % len(__marker_rotation__)],
                markersize = 15.0)
    if prescribed_legend is None:
        plt.legend(legend_entries, loc = leg_loc, ncol = legend_cols,
                   fontsize = legend_fontsize)
    else:
        plt.legend(prescribed_legend, loc = leg_loc, ncol = legend_cols,
                   fontsize = legend_fontsize)
    if xlabel is None:
        plt.xlabel(r'Support size')
    else:
        plt.xlabel(xlabel)
    if ylabel is None:
        plt.ylabel(r'Average number of tiles')
    else:
        plt.ylabel(ylabel)
    if title is None:
        plt.title(r'Average number of tiles vs support size')
    else:
        plt.title(title)
    plt.ylim([0.95, 1.1 * np.max(n_avg_supports)])
    if save_as is not None:
        fig.savefig(save_as)
    plt.show()
