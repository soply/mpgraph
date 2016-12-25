# coding: utf8

from timeit import default_timer as timer

import numpy as np
from tabulate import tabulate

from tilingElement import TilingElement


def wrapper_create_tiling(A, y, u_real, beta_min, beta_max, n_sparsity,
                          prior=None, options=None):
    """ Wrapper for the tiling object. Creates the tiling object and directly
    runs the create_tiling method between given beta_min and beta_max.

    Parameters
    ----------
    A : array, shape (n_measurements, n_features)
        Measurement matrix in A(u+v) = y

    y : array, shape (n_measurements)
        Measurement vector in A(u+v) = y.

    u_real : array, shape (n_features)
        Vector u in A(u+v) = y that was used to create the datum y (in addition
        to some disturbance v). As such it can be used to assess the
        characteristics of the found solution u_(alpha, beta) with u_real but
        it shall be noted that the splitting u + v is by no means unique. This
        is only achieved by using regularisation terms to restrict the solution.

    beta_min : Positive real number
        Minimal boundary of beta for which the possible supports of a
        solution u_(alpha, beta) shall be calculated.

    beta_min : Positive real number > beta_min
        Maximal boundary of beta for which the possible supports of a
        solution u_(alpha, beta) shall be calculated.

    n_sparsity : Integer
        As soon as a specific tiling element reaches a support of size
        n_sparsity, the algorithm does not search for more children, thus
        potentially larger supports for such a tiling element. Therefore, this
        number is an upper bound of the supports that will be searched during
        running the algorithm. Note however that it does not necessarily mean
        that we find all supports of size n_sparsity since entries can
        be dropped if we follow a specific solution path.
        Since for 'LARS' mode entries can not drop out of the support, in this
        case we find all supports with support size n_sparsity.

    prior : array, shape (n_features)
        Prior information on the noise/disturbance vector v. If prior is given
        the problem
            1/2||A(u+v)-(y-A*prior)||_2^2+alpha*||u||_1+beta/2*||v-prior||_2^2
        is solved.

    options : dict, keys ...
        Dictionary specifying the options that are passed to the Tiling object.
        Check default_options method of tiling object to check what options can
        be set.

    Return
    ----------
    tiling : object of class Tiling
        Processed tiling object that contains the graph corresponding to
        the support tiling that is caused by the given data.
    """
    tiling = Tiling(A, y, u_real, prior, options)
    tiling.create_tiling(beta_min, beta_max, n_sparsity)
    return tiling


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
        self.root_element.verify_tiling()
        tab = self.tabularise_results()
        print tabulate(tab, headers = tabularised_result_column_descriptor())
        self.root_element.plot_tiling(n_disc = 5)
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

    def find_support_to_supportsize(self, support_size):
        tiling_elements = self.root_element.bds_order()
        return [te for te in tiling_elements.keys() if
                                                len(te.support) == support_size]

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
