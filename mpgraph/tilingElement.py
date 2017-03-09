# coding: utf8
""" Implementation of elements of a tiling as a class. Each object of this class,
    called TilingElement, constitutes a single tile of the tiling, ie. a
    connected area of parameters that leads to the same support and sign
    pattern. """

__author__ = "Timo Klock"

import numpy as np

from create_children.create_children_lars import create_children_lars
from create_children.create_children_lasso import create_children_lasso

class TilingElement(object):
    """ An object of the class TilingElement forms a single tile of the support
    tiling of the solution to the multi-penalty functional

        J_beta,alpha(u,v) = 1/2 || A(u+v) - y||_2^2 + alpha ||u||_1
                            + beta/2 ||v||_2^2.

    A tile is characterized by a connected area in the positive Euclidean plane
    R_+^2 such that all parameters inside this area lead to the same support and
    sign pattern. Forming all tiles, respectively objects of class TilingElement
    for a fixed problem together, yields the full support tiling. These objects
    are represented by the Tiling class, hence an object of the Tiling class is
    usually related to many objects of the TilingElement class.

    A tile usually has two markant parameters pairs: (alpha_min, beta_min) and
    (alpha_max, beta_max). These are (usually) the smallest parameters such that
    the respective support of the tile is reached as well as the largest
    parameters.

    More details on how the meaning of a tile and how we find these tiles for a
    given fixed problems of the above mentioned type, can be found in the code
    docs and in the article [1].

    Sources
    ------------------
    [1]
    """

    def __init__(self, alpha_min, alpha_max, beta_min, beta_max, support,
                 sign_pattern, parents, A, y, svdAAt_U, svdAAt_S, options):
        """ Constructor for the TilingElement class.

        Parameters
        -----------
        alpha_min : python float
            Regularisation parameter alpha for the smallest parameter pair.

        alpha_max : python float
            Regularisation parameter alpha for the largest parameter pair.

        beta_min : python float
            Regularisation parameter beta for the smallest parameter pair.

        beta_max : python float
            Regularisation parameter beta for the largest parameter pair.

        support : np.array of integers, shape (n_support)
            Support of solution u_beta,alpha related to this tile.

        sign_pattern : np.array of +,- 1, shape (n_support)
            Sign pattern of solution u_beta,alpha related to this tile.

        parents : list of tuples of the form (object of class TilingElement,
            float, float).
            Contains the parents of this tile in the support tiling (first entry),
            as well as the beta-range inside which we can reach this tile from
            the respective parent (2nd entry -> lower beta, 3rd entry -> higher
            beta)

        A : array, shape (n_measurements, n_features)
            Measurement matrix in A(u+v) = y

        y : array, shape (n_measurements)
            Measurement vector in A(u+v) = y.

        svdU : array, shape (n_measurements, n_measurements)
            Matrix U of the singular value decomposition of A*A^T.

        svdS : array, shape (n_measurements)
            Array S with singular values of singular value decomposition of A*A^T.

        options : python dict objects
            Specifies the options for the run. See constructor to see which
            options can be specified.

        Remarks
        ---------------
        The problem data is only given by reference, ie. A, y, svdU, svdS points
        to the same object among all tiles and the overlying tiling object. The
        SVD matrices are heavily used in the find_children procedure and are
        thus important to have here (speed-up!).
        """
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.support = support
        self.sign_pattern = sign_pattern
        self.parents = parents
        self.children = []
        # References to global data matrices and data
        self.A = A
        self.y = y
        self.svdAAt_U = svdAAt_U
        self.svdAAt_S = svdAAt_S
        self.options = options
        self.identifier = None

    def __repr__(self):
        """ Overwriting the build-in representation method. """
        return ("TilingElement(Support: {0}" +
                " Parameter range: {1})").format(self.support,
                                                 [(self.alpha_min, self.beta_min),
                                                  (self.alpha_max, self.beta_max)])

    def identify_as(self, identifier):
        """ Method to add identifier as attribute self.identifier to a particular
        tilingelement.

        Parameters
        -----------
        identifier : Integer number
            Identifier of tiling element
        """

        self.identifier = identifier

    def get_root(self):
        """ Returns the root node of the tiling this element belongs to.

        Returns
        --------------
        The root node that belongs to the tiling connected to this tiling
        element.
        """
        curr_node = self
        while curr_node.parents is not None:
            curr_node = curr_node.parents[0][0]
        return curr_node

    def index_in_many_predecessors(self, beta_min, beta_max, n_predecessors,
                                   index):
        """ Checks if a specific entry is in all n predecessors of this node
        provided that the predecessor has overlap with the given parameter
        region (beta_min, beta_max).

        Parameters
        -----------
        beta_min : python float
            Lower bound for parameter region

        beta_max : python float
            Upper bound for parameter region

        n_predecessors : python Integer
            Number of predecessors to check

        index : python Integer
            Index to check

        Returns
        ------------
        True if the index is in all n predecessors, false otherwise.
        """
        if index not in self.support:
            return False
        elif n_predecessors == 0:
            return True
        else:
            found_wrong_predecessor = False
            for parent in self.parents:
                if (parent[1] - beta_max < -1e-14 and
                        beta_min - parent[2] < -1e-14) and not \
                        parent[0].index_in_many_predecessors(beta_min,
                            beta_max, n_predecessors - 1, index):
                    return False
            return True

    def find_children(self, beta_min=None, beta_max=None):
        """ Method to find children of the given tiling element inside the given
        range (beta_min, beta_max). This method calls, according to the specified
        mode, a method to find children. Afterwards the children are added
        tentatively to this tiling element, therefore an additional tiling
        merging operation should be called after this function has been used.
        To read about the different methods of finding children, we refer to the
        respective files in the 'create_children' folder.

        Parameters
        ------------
        beta_min : python double
            Lower boundary of the range of beta's

        beta_max : python double
            Upper boundary of the range of beta's

        Returns
        -----------
        Adds the new children (ie. the related, newly created tiling elements)
        tentatively to this tiling element. Returns a list with all created
        tiling elements.

        Remarks
        -----------
        Available modes:
            -'LASSO': Using the Lasso-path algorithm to create the tiling
            -'LARS': Using the LAR(S) algorithm to create the tiling, ie. the
                     Lasso-path algorithm with neglecting dropping of indices.
        """
        if beta_min is None:
            beta_min = self.beta_min
        if beta_max is None:
            beta_max = self.beta_max
        if self.options["mode"] == "LARS" or self.parents is None:
            children = create_children_lars(self, beta_min, beta_max)
        elif self.options["mode"] == "LASSO":
            children = create_children_lasso(self, beta_min, beta_max)
        elif self.options["mode"] == "TEST":
            # For debugging and testing only
            children = self.options["test_iterator"].next()
            children.sort(key=lambda x: x[0][1])
        else:
            raise NotImplementedError("Mode not implemented.")
        new_tes = []
        for child in children:
            new_tes.append(self.add_child(child[0][0], child[1][0], child[0][1],
                                          child[1][1], child[2], child[3]))
        self.sort_children()
        return new_tes

    def child_to_beta(self, beta):
        """ Returns the child element for the given beta.

        Parameters
        -------------
        beta : Positive, real number with self.beta_min <= beta <= self.beta_max.

        Returns
        ----------
        Successor/child of this tiling element for a specific beta.
        """
        if len(self.children) > 0:
            return [child for child in self.children if child[1] <= beta \
                                                    and child[2] >= beta][0][0]
        else:
            return None

    def shares_support_with(self, tilingelement):
        """ Checks whether the current tiling element shares the support and
        sign pattern with the given tiling element.

        Parameters
        -----------
        tilingelement : object of class TilingElement
            Tiling element to compare to.

        Returns
        -----------
        True if both tiling elements have the same support and sign pattern,
        false otherwise.
        """
        return np.array_equal(tilingelement.support, self.support) and \
            np.array_equal(tilingelement.sign_pattern, self.sign_pattern)

    def can_be_merged_with(self, tilingelement):
        """ Checks whether the current tiling element can be merged with the
        given tiling element. Two tiling elements can be merged if they share
        the same support, sign pattern and have consecutive parameter regions.

        Parameters
        -----------
        tilingelement : object of class TilingElement
            Tiling element to compare to.

        Returns
        -----------
        True if both tiling elements can be merged, false otherwise.
        """
        return self.shares_support_with(tilingelement) and \
            ((np.abs(self.alpha_max - tilingelement.alpha_min) /
              self.alpha_max < 1e-4 and
              self.beta_max == tilingelement.beta_min) or
             (np.abs(self.alpha_min - tilingelement.alpha_max) /
              self.alpha_min < 1e-4 and
              self.beta_min == tilingelement.beta_max))

    def add_child(self, alpha_min, alpha_max, beta_min, beta_max, support,
                  signum):
        """ Creates a new tiling element from the given arguments and adds it to
        the list of children of this tiling element. Note that the self node is
        also automatically assigned as a parent of the newly created node.

        Parameters
        ------------
        alpha_min : float
            Minimum value for regularisation parameter alpha (corresponding
            to beta_min) to reach the new child.

        alpha_max : float
            Maximum value for regularisation parameter alpha (corresponding
            to beta_max) to reach the new child.

        alpha_min : float
            Minimum value for regularisation parameter beta (corresponding
            to alpha_min) to reach the new child.

        alpha_max : float
            Maximum value for regularisation parameter beta (corresponding
            to alpha_max) to reach the new child.

        support : np.array (shape n_support)
            Contains the indices corresponding to the support of the new child.

        signum : np.array (shape n_support)
            {-+1}^n_support representing the sign pattern of the new child.

        Returns
        ------------
        An object of class TilingElement that is the newly created child.
        """
        te = TilingElement(alpha_min, alpha_max, beta_min, beta_max, support,
                           signum, [[self, beta_min, beta_max]], self.A,
                           self.y, self.svdAAt_U, self.svdAAt_S, self.options)
        self.children.append([te, beta_min, beta_max])
        return te

    def sort_children(self):
        """ Sorts the children of the current tiling element in place by the
        minimum beta for which a child element can be reached from the current
        element.

        Remark
        ----------
        Since for each beta there should only be one distinct child element, the
        operation is well defined.
        """
        self.children.sort(key=lambda x: x[1])

    def sort_parents(self):
        """ Sorts the parents of the current tiling element in place by the
        minimum beta for which the current element can be reached from a
        respective parent element.

        Remark
        ----------
        Since the current node has a distinct parent for each beta in
        [self.beta_min, self.beta_max], the operation is well defined.
        """
        self.parents.sort(key=lambda x: x[1])

    def uniquefy_children(self):
        """ Uniquefies the children of the current tiling element by deleting
        children that share the same tiling element and connected parameter regions.
        Such situations can occur after merging and searching for subsitute
        children via find_children in a part of the actual beta range of this
        node. As such, it can also be called an 'inner-layer' merge operation.

        Remark
        ---------------
        The implementation assumes and works only in case the children of this
        tiling element are sorted by the minimum/maximum beta for which a
        specific child can be reached from this tiling element. Hence, children
        with connected parameter regions are successors in self.children.

        FIXME: Add doc example
        """
        ctr = 1
        n_children = len(self.children)
        while ctr < len(self.children):
            if self.children[ctr - 1][0] == self.children[ctr][0]:
                self.children[ctr - 1][2] = self.children[ctr][2]
                del self.children[ctr]
                n_children -= 1
            else:
                ctr += 1

    def uniquefy_parents(self):
        """ Uniquefies the parents of the current tiling element by deleting
        parents that share the same tiling element and connected parameter
        regions.

        Remark
        ---------------
        The implementation assumes and works only in case the parents of this
        tiling element are sorted by the minimum/maximum beta for which a
        specific parent can be reached from this tiling element. Hence, parents
        with connected parameter regions are successors in self.parents.

        FIXME: When do such situations occur exactly? Add doc example
        """
        ctr = 1
        n_parents = len(self.parents)
        while ctr < len(self.parents):
            if self.parents[ctr - 1][0] == self.parents[ctr][0]:
                self.parents[ctr - 1][2] = self.parents[ctr][2]
                del self.parents[ctr]
                n_parents -= 1
            else:
                ctr += 1

    def replace_child(self, child_to_replace, replacement):
        """ Replaces tiling element in the children of this tiling element by
        the given replacement.

        Parameters
        --------------
        child_to_replace: object of class TilingElement
            Tiling element that needs to be replaced. Should be available in
            self.children.

        replacement: object of class TilingElement
            Tiling element that replaces the child.

        Remarks
        ---------------
        Assumes that the tiling element 'child_to_replace' is in self.children.
        Otherwise it throws a RuntimeError.
        """
        ctr = 0
        while ctr < len(self.children):
            if self.children[ctr][0] != child_to_replace:
                ctr += 1
            else:
                self.children[ctr][0] = replacement
                return 1
        else:
            raise RuntimeError(("Could not find child {0} in the children of"+\
                                " child {1} and thus could not replace it"+\
                                " with child {2}").format(self, child_to_replace,
                                                          replacement))

    def replace_parent(self, parent_to_replace, replacement):
        """ Replaces tiling element in the parents of this tiling element by
        the given replacement.

        Parameters
        --------------
        parent_to_replace: object of class TilingElement
            Tiling element that needs to be replaced. Should be available in
            self.parents.

        replacement: object of class TilingElement
            Tiling element that replaces the parent.

        Remarks
        ---------------
        Assumes that the tiling element 'parent_to_replace' is in self.parents.
        Otherwise it throws a RuntimeError.
        """
        ctr = 0
        while ctr < len(self.parents):
            if self.parents[ctr][0] != parent_to_replace:
                ctr += 1
            else:
                self.parents[ctr][0] = replacement
                return 1
        else:
            raise RuntimeError(("Could not find child {0} in the parents of"+\
                                " child {1} and thus could not replace it"+\
                                " with child {2}").format(self, parent_to_replace,
                                                          replacement))

    def oldest_child(self):
        """ Returns the oldest child of this tiling element. The oldest child
        is the child that can currently be reached for the smallest beta
        (self.beta_min if list of children of this element is already complete).

        Returns
        ---------
        Object of class tiling element corresponding to the oldest child; or
        None if this tiling element has to children.

        Remark
        ----------
        Assumes the children of this node are sorted (by self.sort_children()).
        """
        if len(self.children) > 0:
            return self.children[0][0]
        else:
            return None

    def youngest_child(self):
        """ Returns the youngest child of this tiling element. The youngest
        child is the child that can currently be reached for the largest beta
        (self.beta_max if list of children of this element is already complete).

        Returns
        ---------
        Object of class tiling element corresponding to the youngest child; or
        None if this tiling element has no children.

        Remark
        ----------
        Assumes the children of this node are sorted (by self.sort_children()).
        """
        if len(self.children) > 0:
            return self.children[-1][0]
        else:
            return None

    def oldest_parent(self):
        """ Returns the oldest parent of this tiling element. The oldest parent
        is the parent that can currently be reached for the smallest beta
        (self.beta_min if list of parents of this element is already complete).

        Returns
        ---------
        Object of class tiling element corresponding to the oldest parent; or
        None if this tiling element has no parents (only root node).

        Remark
        ----------
        Assumes the parents of this node are sorted (by self.sort_parents()).
        """
        if self.parents is not None and len(self.parents) > 0:
            return self.parents[0][0]
        else:
            return None

    def youngest_parent(self):
        """ Returns the youngest parent of this tiling element. The youngest
        parent is the parent that can currently be reached for the largest beta
        (self.beta_max if list of parents of this element is already complete).

        Returns
        ---------
        Object of class tiling element corresponding to the youngest parent; or
        None if this tiling element has no parents (only root node).

        Remark
        ----------
        Assumes the parents of this node are sorted (by self.sort_parents()).
        """
        if self.parents is not None and len(self.parents) > 0:
            return self.parents[-1][0]
        else:
            return None

    def find_next_older_neighbor(self, child):
        """ Returns the next older neighbor of the given child outgoing from the
        parent that is given by self. The next older neighbor is defined as the
        tiling element that can be reached from the succeeding beta region of
        the childrens beta region.

        Parameters
        ---------
        child : python object of class TilingElement
            Child to which we search the next older neighbor, outgoing from
            the shared parent self.

        Returns
        ---------
        Object of class TilingElement that is the next older neighbor; or throws
        a RuntimeError if no suceeding neighbor of child in the children of self
        was found.

        Remark
        ----------
        -Assumes the children of self are sorted (by self.sort_children()).
        -Throws exception in case of failure.
        """
        ctr = 1
        n_children = len(self.children)
        while ctr < n_children:
            if self.children[ctr][0] == child:
                return self.children[ctr - 1][0]
            ctr = ctr + 1
        else:
            raise RuntimeError(("Could not find next older neighbor of " +
                                "{0} in children of node {1}.").format(child,
                                                                       self))

    def find_next_younger_neighbor(self, child):
        """ Returns the next younger neighbor of the given child outgoing from
        the parent that is given by self. The next younger neighbor is defined
        as the tiling element that can be reached from the preceding beta
        region of the childrens beta region.

        Parameters
        ---------
        child : python object of class TilingElement
            Child to which we search the next younger neighbor, outgoing from
            the shared parent self.

        Returns
        ---------
        Object of class TilingElement that is the next younger neighbor; or
        throws a RuntimeError if no preceding neighbor of child in the children
        of self was found.

        Remark
        ----------
        -Assumes the children of self are sorted (by self.sort_children()).
        -Throws exception in case of failure.
        """
        ctr = 0
        n_children = len(self.children)
        while ctr + 1 < n_children:
            if self.children[ctr][0] == child:
                return self.children[ctr + 1][0]
            ctr = ctr + 1
        else:
            raise RuntimeError(("Could not find next younger neighbor of " +
                                "{0} in children of node {1}.").format(child,
                                                                       self))

    def find_left_merge_candidate(self):
        """ Method to find a left merging candidate for the given tiling element
        in the related tiling. Left hereby means that the minimal parameters of
        self should be equal to the maximal parameters of the merging candidate.

        The procedure works as follows:

            1) Follow the oldest parent path in the tiling as long as the
            parent's oldest child equals respective source node. If a parent
            with an older child can be found, save this parent and the
            next older neighbor of the source node. Otherwise return None.

            2) From the next older neighbor of the respective source node, trace
            the youngest child path until we can merge a tiling element with the
            self node. If none can be found return Null, otherwise return the
            merging partner.

        Returns
        -------------
        If a merging partner could be found, this tiling element is returned as
        a python object of class TilingElement. Otherwise it returns None.
        """
        assert len(self.children) == 0 and len(self.parents) == 1
        current_node = self
        while current_node.oldest_parent() is not None and \
                current_node.oldest_parent().oldest_child() == current_node:
            current_node = current_node.oldest_parent()
        # Reaching the root without finding younger children -> self is node of
        # the leftmost's part of the graph.
        if current_node.oldest_parent() is None:
            return None
        # Get parent
        parent = current_node.oldest_parent()
        # Get next older sibling
        current_node = parent.find_next_older_neighbor(current_node)
        # Find matching support and sign pattern
        while not current_node.can_be_merged_with(self) and \
                len(current_node.children) > 0:
            current_node = current_node.youngest_child()
        if current_node is not None and current_node.can_be_merged_with(self):
            return current_node
        else:
            return None

    def find_right_merge_candidate(self):
        """ Method to find a right merging candidate for the given tiling element
        in the related tiling. Right hereby means that the maximal parameters of
        self should be equal to the minimal parameters of the merging candidate.

        The procedure works as follows:

            1) Follow the youngest parent path in the tiling as long as the
            parent's youngest child equals respective source node. If a parent
            with a younger child can be found, save this parent and the
            next younger neighbor of the source node. Otherwise return None.

            2) From the next younger neighbor of the respective source node,
            trace the oldest child path until we can merge a tiling element
            with the self node. If none can be found return Null, otherwise
            return the merging partner.

        Returns
        -------------
        If a merging partner could be found, this tiling element is returned as
        a python object of class TilingElement. Otherwise it returns None.
        """
        assert len(self.children) == 0 and len(self.parents) == 1
        current_node = self
        while current_node.youngest_parent() is not None and \
                current_node.youngest_parent().youngest_child() == current_node:
            current_node = current_node.youngest_parent()
        # Reaching the root without finding younger children -> self is node of
        # the leftmost's part of the graph.
        if current_node.youngest_parent() is None:
            return None
        # Get parent
        parent = current_node.youngest_parent()
        # Get next older sibling
        current_node = parent.find_next_younger_neighbor(current_node)
        # Find matching support and sign pattern
        while not current_node.can_be_merged_with(self) and \
                len(current_node.children) > 0:
            current_node = current_node.oldest_child()
        if current_node is not None and current_node.can_be_merged_with(self):
            return current_node
        else:
            return None

    @staticmethod
    def base_region(beta_min, beta_max, A, y, svdU, svdS, options):
        """ Creates and returns the root tiling element object. Note that alpha
        min and alpha max are initialised randomly with 100, but this is not
        influential to the results of the algorithm.

        Parameters
        ------------
        beta_min : float
            Minimal beta of root_element, specifies the minimum of all betas
            considered in a following tiling creation.

        beta_max : float
            Maximal beta of root_element, specifies the maximum of all betas
            considered in a following tiling creation.

        A : array, shape (n_measurements, n_features)
            The sensing/sampling matrix A.

        y : array, shape (n_measurements)
            The vector of measurements

        svdU : array, shape (n_measurements, n_measurements)
            Matrix U of the singular value decomposition of A*A^T.

        svdS : array, shape (n_measurements)
            Array S with singular values of singular value decomposition of A*A^T.

        options : python dict objects
            Specifies the options for the run. See constructor to see which
            options can be specified.

        Returns
        ------------
        Object of class TilingElement that serves as a root object, ie. has
        empty support and sign pattern, dummy initialised alpha_min and
        alpha_max, and the user-specified beta_range as its parameters.
        """
        return TilingElement(100.0, 100.0, beta_min, beta_max,
                             np.array([]).astype("uint32"),
                             np.array([]).astype("int32"), None, A, y, svdU,
                             svdS, options)

    def bds_order(self):
        """ Traverses through the whole graph, starting from this current node,
        in a breadth-deep-search type manner. While doing that, stores all
        tiling elements that can be found as keys of a python dictionary and
        assigns as a corresponding value the layer. This integer defines how
        many Lasso path steps from the given tiling element are (at least!)
        necessary to reach the respective tiling element in keys().

        Returns
        ----------
        python dict object with tiling elements as keys and the corresponding
        minimally necessary number of Lasso path steps to reach it from the
        given source node (self) as the value.
        """
        elements_in_bds_order = {self: 0}
        element_queue = [self]
        while len(element_queue) > 0:
            element = element_queue.pop(0)
            for child in element.children:
                if child[0] not in elements_in_bds_order:
                    elements_in_bds_order[child[0]] = \
                        elements_in_bds_order[element] + 1
                    element_queue.append(child[0])
        return elements_in_bds_order
