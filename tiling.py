# coding: utf8

from timeit import default_timer as timer

import numpy as np
from tabulate import tabulate

from tilingElement import TilingElement


def wrapper_create_tiling(A, y, u_real, beta_min, beta_max, n_layers,
                          prior=None, options=None):
    tiling = Tiling(A, y, u_real, prior, options)
    layers = tiling.create_tiling(beta_min, beta_max, n_layers)
    return layers, tiling


class Tiling(object):
    """ Doc string """

    def __init__(self, A, y, u_real, prior=None, options=None):
        """ Contructor doc string """
        if options is None:
            options = self.default_options()
        else:
            options = dict(self.default_options().items() + options.items())
        # If prior knowledge on v is given
        if prior is not None:
            y = y - A.dot(prior)
        self.A = A
        self.y = y
        self.u_real = u_real
        self.prior = prior
        self.options = options
        self.start_time = timer()
        self.elapsed_time = 0
        U, S, UT = np.linalg.svd(A.dot(A.T))
        self.svdU = U  # Pre-calc since needed very often
        self.svdS = S  # Pre-calc since needed very often
        self.root_element = None

    def create_tiling(self, beta_min, beta_max, n_sparsity, options=None):
        # Override options if desired
        print "Beginning tiling creation..."
        if options is not None:
            self.options = dict(self.options.items() + options.items())
        self.root_element = TilingElement.base_region(beta_min, beta_max, self.A,
                                                      self.y, self.svdU, self.svdS,
                                                      self.options)
        stack = [self.root_element]
        while len(stack) > 0:
            current_element = stack.pop(0)
            children = current_element.find_children()
            uncompleted_children, children_for_stack = \
                TilingElement.merge_new_children(children)

            stack.extend(list(filter_children_sparsity(children_for_stack,
                                                       n_sparsity)))
            while len(uncompleted_children) > 0:
                uncomp_child, beta_min, beta_max = uncompleted_children.pop(0)
                children = uncomp_child.find_children(beta_min, beta_max)
                tmp_uncomp_children, children_for_stack = \
                    TilingElement.merge_new_children(children)
                uncompleted_children.extend(tmp_uncomp_children)
                stack.extend(list(filter_children_sparsity(children_for_stack,
                                                           n_sparsity)))
        print "Finished tiling creation..."
        tab = self.tabularise_results()
        print tabulate(tab, headers = tabularised_result_column_descriptor())
        import pdb
        pdb.set_trace()


    def default_options(self):
        """ Default option setting """
        return {
            # Verbosity levels: 0: Results only,
            #                   1: Summary Tables,
            #                   2: Everything (debugging)
            "verbose": 2,
            "mode": "LARS",
            # Minimiser to find intersection between two curves
            "env_minimiser": "scipy_brentq",
            # Processes spawned if multi-processing shall be used
            "max_processes": 1,
        }

    def tabularise_results(self):
        tiling_elements = self.root_element.bds_order()
        tiling_elements = list(tiling_elements.iteritems())
        tiling_elements.sort(key=lambda x: (len(x[0].support), x[0].alpha_min))
        results = np.zeros((len(tiling_elements), 6))
        real_support = np.where(self.u_real)[0]
        for (i, (te, layer)) in enumerate(tiling_elements):
            results[i,0] = te.alpha_min
            results[i,1] = te.beta_min
            results[i,2] = te.alpha_max
            results[i,3] = te.beta_max
            results[i,4] = len(te.support)
            results[i,5] = len(np.setdiff1d(te.support, real_support)) + \
                            len(np.setdiff1d(real_support, te.support))
        return results


def filter_children_sparsity(children, sparsity_bound):
    for child in children:
        if len(child.support) < sparsity_bound:
            yield child

def tabularised_result_column_descriptor():
    """ Return a list with strings that explain the columns of the results_mp
    and result_sp meaning. """
    return [
        "alpha_min",
        "beta_min",
        "alpha_max",
        "beta_max",
        "#Supp",
        "Sym. Dif.",
    ]
