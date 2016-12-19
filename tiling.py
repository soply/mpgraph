#coding: utf8

from timeit import default_timer as timer

import numpy as np

from tilingElement import TilingElement

def wrapper_create_tiling(A, y, u_real, beta_min, beta_max, n_layers,
                          prior = None, options = None):
    tiling = Tiling(A, y, u_real, prior, options)
    layers = tiling.create_tiling(beta_min, beta_max, n_layers)
    return layers, tiling

class Tiling(object):
    """ Doc string """

    def __init__(self, A, y, u_real, prior = None, options = None):
        """ Contructor doc string """
        if options is None:
            options = self._default_options()
        else:
            options = dict(self._default_options().items() + options.items())
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
        self.svdU = U # Pre-calc since needed very often
        self.svdS = S # Pre-calc since needed very often
        self.root_element = None

    def create_tiling(self, beta_min, beta_max, n_sparsity, options = None):
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
        self.root_element.plot_graph()

    def _default_options(self):
        """ Default option setting """
        return {
            # Verbosity levels: 0: Results only,
            #                   1: Summary Tables,
            #                   2: Everything (debugging)
            "verbose" : 2,
            "mode" : "LARS",
            # Minimiser to find intersection between two curves
            "env_minimiser" : "scipy_brentq",
            # Processes spawned if multi-processing shall be used
            "max_processes" : 1,
        }

def filter_children_sparsity(children, sparsity_bound):
    for child in children:
        if len(child.support) < sparsity_bound:
            yield child
