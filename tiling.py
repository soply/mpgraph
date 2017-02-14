# coding: utf8

from timeit import default_timer as timer

import numpy as np
from tabulate import tabulate

from mp_utils import approximate_solve_mp_fixed_support
from tilingElement import TilingElement
from tilingVerification import plot_tiling, plot_tiling_graph, verify_tiling


def wrapper_create_tiling(A, y, beta_min, beta_max, n_sparsity, prior=None,
                          options=None):
    """ Wrapper for the tiling object. Creates the tiling object and directly
    runs the create_tiling method between given beta_min and beta_max.

    Parameters
    ----------
    A : array, shape (n_measurements, n_features)
        Measurement matrix in A(u+v) = y

    y : array, shape (n_measurements)
        Measurement vector in A(u+v) = y.

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
    tiling = Tiling(A, y, prior, options)
    tiling.create_tiling(beta_min, beta_max, n_sparsity)
    return tiling


class Tiling(object):
    """ Doc string """

    def __init__(self, A, y, prior=None, options=None):
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
        self.prior = prior
        self.options = options
        starttime_svd = timer()
        U, S, UT = np.linalg.svd(A.dot(A.T))
        self.svdU = U  # Pre-calc since needed very often
        self.svdS = S  # Pre-calc since needed very often
        self.elapsed_time_svd = timer() - starttime_svd
        self.elapsed_time_tiling = 0
        self.root_element = None

    def create_tiling(self, beta_min, beta_max, n_sparsity, options=None):
        # Override options if desired
        if options is not None:
            self.options = dict(self.options.items() + options.items())
        print "Beginning tiling creation..."
        starttime_tiling = timer()
        self.root_element = TilingElement.base_region(beta_min, beta_max, self.A,
                                                      self.y, self.svdU, self.svdS,
                                                      self.options)
        stack = [self.root_element]
        while len(stack) > 0:
            if self.options['verbose'] >= 1:
                print "Current stack size: {0}".format(len(stack))
                print "Minimum support length on stack: {0}".format(
                                                        len(stack[0].support))
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
        self.assign_identifiers_to_elements()
        self.elapsed_time_tiling = timer() - starttime_tiling
        print "Finished tiling creation..."
        if self.options.get('print_summary', False):
            tab = self.tabularise_results()
            print tabulate(tab, headers=["Identifier", "alpha_min", "beta_min",
                    "alpha_max", "beta_max", "#Supp", "Sym. Dif."])

    def default_options(self):
        """ Returns a default option setting. Each option that is not overriden
        by user-specified options will be used in the tiling creation.

        Option dictonary-key | Description
        ------------------------------------------------------------------------
        verbose | Verbosity level determines how much print outs we have. The
                  higher the integer value is chosen, the more print-outs will
                  happen.
        mode | Determines the mode to find children of a certain tiling element.
               Currently available are:
               'LARS' : Performs a 'least angle regression'-step to find new
                        children of a specific tiling element.
               'LASSO' : Performs a 'lasso path'-step to find new children of
                         a specific tiling element.
        env_minimiser | The minimiser that is used to minimise respective
                        root-finding/minimisation problems to calculate the
                        envelope in the chosen find_children method.
        print_summary | If True, the results of the tiling creation are printed
                        in tabularised form to the console. Otherwise, no
                        print-outs will be done at the end.

        Returns
        ----------------
        Default options settings as a python dictionary.
        """
        return {
            "verbose": 2,
            "mode": "LARS",
            "env_minimiser": "scipy_brentq",
            "print_summary" : True
        }

    def find_support_to_supportsize(self, support_size):
        """ Method to find all tiling elements in a calculated tiling that have
        a support with size equal to a specified support size. Note that the
        tiling has to be created prior to calling this method.

        Parameters
        ---------------
        support_size : Integer
            Support size of tiling elements that we want to find in the tiling.

        Returns
        ---------------
        python list of objects of class TilingElement where all tiling elements
        have a support equal to the given support size. Candidates are all
        tiling elements that have been calculated before.
        """
        tiling_elements = self.root_element.bds_order()
        return [te for te in tiling_elements.keys() if
                                                len(te.support) == support_size]

    def find_supportpath_to_beta(self, beta):
        """ Method to find all supports and sign patterns for a fixed beta in
        the given tiling. Note that the tiling has to be calculated prior to
        calling this method.

        Parameters
        ---------------
        beta : Positive, real number
            beta for which we want to return the support and sign-pattern path.
            Note that the beta must be satisfy
            self.root_element.beta_min <= beta <= self.root_element.beta_max.


        Returns
        ---------------
        Tuple with two python lists. The first list contains all supports that
        can be found in the tiling for a fixed beta. The second list contains
        all sign patterns that belong to the supports in the first list.
        """
        assert self.root_element is not None
        assert self.root_element.beta_min <= beta and \
               self.root_element.beta_max >= beta
        current_element = self.root_element
        supports = []
        sign_patterns = []
        while current_element is not None:
            supports.append(current_element.support)
            sign_patterns.append(current_element.sign_pattern)
            current_element = current_element.child_to_beta(beta)
        return supports, sign_patterns

    def tabularise_results(self, u_real_for_comparison = None):
        """ Method to tabulise the tiling results. Each row belongs to a single
        tiling element and each row contains the following options about a
        specific tiling element:
        0 - identifier : Integer that uniquely identifies a specific tiling
                         element of this tiling.
        1 - alpha_min : Minimal regularisation parameter alpha for which we can
                        reach a specific support.
                        The corresponding beta regularisation parameter is given
                        in the next entry.
        2 - beta_min : Minimal regularisation parameter beta for which we can
                       reach a specific support.
        3 - alpha_max : Maximal regularisation parameter alpha for which we can
                        reach a specific support.
                        The corresponding beta regularisation parameter is given
                        in the next entry.
        4 - beta_max : Maximal regularisation parameter beta for which we can
                       reach a specific support.
        5 - Support size : Support size of the support of a tiling element.
        Optional :
        6 - Symmetric difference : Symmetric difference to a target support
                                   where the target support is specified is
                                   given by argument 'u_real_for_comparison'.

        Parameters
        --------------
        u_real_for_comparison (optional) : array, shape (n_features)
            Optional argument to provide a target support that should be used
            for comparison with results contained in the tiling.

        Returns
        ----------------
        array of shape (n_tiling_elements, 5 (6)) containing the tabularised
        results. If 'u_real_for_comparison' is provided the columnsize is 6,
        otherwise the columnsize is 5. """
        tiling_elements = self.root_element.bds_order()
        tiling_elements = list(tiling_elements.iteritems())
        tiling_elements.sort(key=lambda x: (len(x[0].support), x[0].alpha_min))
        results = np.zeros((len(tiling_elements), 7))
        if u_real_for_comparison is not None:
            real_support = np.where(u_real_for_comparison)[0]
        else:
            real_support = np.zeros(self.A.shape[1])
        for (i, (te, layer)) in enumerate(tiling_elements):
            results[i,0] = te.identifier
            results[i,1] = te.alpha_min
            results[i,2] = te.beta_min
            results[i,3] = te.alpha_max
            results[i,4] = te.beta_max
            results[i,5] = len(te.support)
            results[i,6] = len(np.setdiff1d(te.support, real_support)) + \
                            len(np.setdiff1d(real_support, te.support))
        return results

    def show_table(self, u_real_for_comparison = None):
        """ Show the summary table after creating the tiling. Can be given the
        real solution to compare and see symmetric differences.

        Parameters
        ------------
        u_real_for_comparison : np.array, shape (n_features)
            The signal that was used in generating the problem data (or
            something else to compare the found collected to).
        """
        tab = self.tabularise_results(u_real_for_comparison =
                                            u_real_for_comparison)
        print tabulate(tab, headers=["Identifier", "alpha_min", "beta_min",
                "alpha_max", "beta_max", "#Supp", "Sym. Dif."])

    def plot_tiling(self, n_disc = 3):
        """ Wrapper for plotting the reconstructed tiling. Calls method from
        tilingVerification.py on the root element.
        """
        plot_tiling(self.root_element, n_disc = n_disc)

    def plot_tiling_graph(self, y_mode='layered'):
        """ Wrapper for plotting the graph corresponding to a tiling. Calls
        method from tilingVerification.py on the root element.
        """
        plot_tiling_graph(self.root_element, y_mode)

    def verify_tiling(self):
        """ Wrapper for verifying a reconstructed tiling. Calls method from
        tilingVerification.py on the root element.
        """
        verify_tiling(self.root_element)

    def get_solution_to_element(self, identifier):
        """ Retrieve the solution to a specific tiling element, specified by its
        identifier, where the solution is calculated via a least-squares
        regression on the respective fixed support without additional
        regularization. Note that, if there is a lot of measurement noise on y
        that is not covered by v, the resulting u_I does fit this measurement
        noise and hence maybe not useful.

        Parameters
        ------------
        identifier : Integer
            Identifier of the tile.

        Returns
        -----------
        Returns tuple (u_I, v_I) where u_I is the least squares regression on
        the fixed support (wo regularisation) and v_I is the least squares
        regression of the discrepancy (wo regularisation).        
        """
        te = self.get_tiling_element(identifier)
        u_I, v_I = approximate_solve_mp_fixed_support(te.support, self.A,
                                                      self.y)
        return u_I, v_I

    def get_tiling_element(self, identifier):
        """ Retrieve a specific tiling element by the identifier (listed in
        summary tables).

        Parameters
        ------------
        identifier : Integer
            Identifier of the tile.

        Returns
        ------------
        Object of class TilingElement.
        """
        tes_in_bds = self.root_element.bds_order()
        te = [te for te in tes_in_bds.keys() if te.identifier == identifier][0]
        return te

    def assign_identifiers_to_elements(self):
        """ Assigns unique identifiers to all tiling elements that belong to
        this tile. The order/number itself does not have any meaning, it is
        just important to distinguish and retrieve them.
        """
        tilingelements_in_bds = self.root_element.bds_order()
        identifier = 0
        for element in tilingelements_in_bds.keys():
            element.identify_as(identifier)
            identifier += 1

    def intersection_support_per_layer(self):
        """ Performs a BDS search of the whole tiling, giving all tiling
        elements as well as the minimal number of find_children steps to reach
        these tilings. The latter number is considered as the layer. Then
        a dictionary with keys() given as all possible layers and the
        corresponding value is the intersection of all supports that belong to
        a specific layer.

        Returns
        ------------
        python dict object with different layers as keys and corresponding
        intersected supports as values.
        """
        tilingelements_in_bds = self.root_element.bds_order()
        del tilingelements_in_bds[self.root_element] # Remove root element
        supports = {}
        for layer in set(tilingelements_in_bds.values()):
            tes_to_layer = [te for te in tilingelements_in_bds
                                        if tilingelements_in_bds[te] == layer]
            current_support = tes_to_layer.pop().support
            while len(tes_to_layer) > 0:
                current_support = np.intersect1d(current_support,
                                                 tes_to_layer.pop().support)
            supports[layer] = current_support
        return supports


def filter_children_sparsity(children, sparsity_bound):
    """ Creates and iterator out of a given list of children that yields only
    children whose support is below a specific size. The size is specified by
    sparsity_bound.

    Parameters
    -----------------
    children : python iterable of objects of class TilingElement
        List of tiling elements that are candidates to be yielded by the
        resulting iterator.

    sparsity_bound : integer
        Upper bound for the support size of tiling elements that we will yield.

    Returns
    ------------------
    Python iterator of tiling elements with supports below a given upper bound.
    """
    for child in children:
        if len(child.support) < sparsity_bound:
            yield child
