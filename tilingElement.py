# coding: utf8
import operator
from itertools import groupby

import matplotlib.pyplot as plt
import numpy as np

from create_children.create_children_lars import (create_children_lars,
                                                  lars_post_process_children)
from create_children.create_children_lasso import (create_children_lasso,
                                                   lasso_children_merge,
                                                   lasso_post_process_children)
from create_children.lasso_path_utils import calc_all_cand
from mp_utils import calc_B_y_beta


class TilingElement(object):

    def __init__(self, alpha_min, alpha_max, beta_min, beta_max, support,
                 sign_pattern, parents, A, y, svdAAt_U, svdAAt_S, options=None):
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

    def __repr__(self):
        return ("TilingElement(Support: {0}" +
                " Parameter range: {1})").format(self.support,
                                                 [(self.alpha_min, self.beta_min),
                                                  (self.alpha_max, self.beta_max)])

    def find_children(self, beta_min=None, beta_max=None):
        if beta_min is None:
            beta_min = self.beta_min
        if beta_max is None:
            beta_max = self.beta_max
        if self.options["mode"] == "LARS" or self.parents is None:
            additional_indices, boundary_parameters, used_signs = \
                create_children_lars(self.support, self.sign_pattern, beta_min,
                                     beta_max,
                                     self.options["env_minimiser"],
                                     self.svdAAt_U, self.svdAAt_S, self.A,
                                     self.y)
            children = lars_post_process_children(additional_indices,
                                                  boundary_parameters,
                                                  used_signs, self.support,
                                                  self.sign_pattern)
            children.sort(key=lambda x: x[0][1])
        elif self.options["mode"] == "LASSO":
            children = []
            # If this tiling element has different parents, we have to iterate
            # through these tilings separately because the recently added
            # component from parent to child changed then.
            for parent in self.parents:
                # Checks for overlap
                if (parent[1] - beta_max < -1e-14 and
                        beta_min - parent[2] < -1e-14):
                    new_index = np.setxor1d(self.support, parent[0].support)
                    assert len(new_index) == 1
                    new_index = new_index[0]
                    additional_indices, boundary_parameters, used_signs = \
                        create_children_lasso(self.support, self.sign_pattern,
                                              np.maximum(beta_min, parent[1]),  # beta_min
                                              np.minimum(beta_max, parent[2]),  # beta_max
                                              self.options["env_minimiser"],
                                              self.svdAAt_U, self.svdAAt_S, self.A,
                                              self.y,
                                              new_index)
                    children.extend(lasso_post_process_children(additional_indices,
                                                                boundary_parameters,
                                                                used_signs, self.support,
                                                                self.sign_pattern))
            children.sort(key=lambda x: x[0][1])
            children = lasso_children_merge(children)

        elif self.options["mode"] == "TEST":
            children = self.options["test_iterator"].next()
            children.sort(key=lambda x: x[0][1])
        else:
            raise NotImplementedError("Mode not implemented.")
        new_tes = []
        for (i, child) in enumerate(children):
            new_tes.append(self.add_child(child[0][0], child[1][0], child[0][1],
                                          child[1][1], child[2], child[3]))
        self.sort_children()
        return new_tes

    def shares_support_with(self, tilingelement):
        """ Checks whether the current tiling element shares the support and
        sign pattern with the given tiling element.

        Parameters
        -----------
        self : object of class TilingElement
            Current node

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
        self : object of class TilingElement
            Current tiling element

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

        Parameters
        ----------
        self : object of class TilingElement
            Current tiling element
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

        Parameters
        ----------
        self : object of class TilingElement
            Current tiling element
        """
        self.parents.sort(key=lambda x: x[1])

    def uniquefy_children(self):
        # Assumes that children are in sorted order!
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
        # Assumes that children are in sorted order!
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
        ctr = 0
        while ctr < len(self.children):
            if self.children[ctr][0] != child_to_replace:
                ctr += 1
            else:
                self.children[ctr][0] = replacement
                return 1
        else:
            raise RuntimeError(("Could not find child {0} in the children of child {1} and" +
                                " thus could not replace it with child {2}").format(self,
                                                                                    child_to_replace,
                                                                                    replacement))

    def replace_parent(self, parent_to_replace, replacement):
        ctr = 0
        while ctr < len(self.parents):
            if self.parents[ctr][0] != parent_to_replace:
                ctr += 1
            else:
                self.parents[ctr][0] = replacement
                return 1
        else:
            raise RuntimeError(("Could not find child {0} in the parents of child {1} and" +
                                " thus could not replace it with child {2}").format(self,
                                                                                    parent_to_replace,
                                                                                    replacement))

    def oldest_child(self):
        if len(self.children) > 0:
            return self.children[0][0]
        else:
            # print ("Tiling element: {0}\nNo children but asking for the" +
            #        " oldest child. ").format(self)
            return None

    def youngest_child(self):
        if len(self.children) > 0:
            return self.children[-1][0]
        else:
            # print ("Tiling element: {0}\nNo children but asking for the" +
            #        " youngest child. ").format(self)
            return None

    def oldest_parent(self):
        if self.parents is not None and len(self.parents) > 0:
            return self.parents[0][0]
        else:
            # print ("Tiling element: {0}\nNo parent but asking for the oldest" +
            #        " parent. ").format(self)
            return None

    def youngest_parent(self):
        if self.parents is not None and len(self.parents) > 0:
            return self.parents[-1][0]
        else:
            # print ("Tiling element: {0}\nNo parent but asking for the" +
            #        " oldest parent. ").format(self)
            return None

    def find_next_older_neighbor(self, child):
        ctr = 1
        n_children = len(self.children)
        while ctr < n_children:
            if self.children[ctr][0] == child:
                return self.children[ctr - 1][0]
            ctr = ctr + 1
        else:
            raise RuntimeError(("Could not find next older neighbor of " +
                                "{0} in children of node {1}.").format(child, self))

    def find_next_younger_neighbor(self, child):
        ctr = 0
        n_children = len(self.children)
        while ctr + 1 < n_children:
            if self.children[ctr][0] == child:
                return self.children[ctr + 1][0]
            ctr = ctr + 1
        else:
            raise RuntimeError(("Could not find next younger neighbor of " +
                                "{0} in children of node {1}.").format(child, self))

    def find_left_merge_candidate(self):
        assert len(self.children) == 0 and len(self.parents) == 1
        current_node = self
        while current_node.oldest_parent() is not None and \
                current_node.oldest_parent().oldest_child() == current_node:
            current_node = current_node.oldest_parent()
        # Reaching the root without finding younger children -> self is node of
        # the leftmost's part of the graph.
        if current_node.oldest_parent() is None:
            # print "(1) Stopping left search operation on {0}".format(self)
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
            # print "(2) Found left merge partner {0} for {1}".format(current_node,
            #                                                         self)
            return current_node

        else:
            # print "(3) Stopping left search operation on {0}".format(self)
            return None

    def find_right_merge_candidate(self):
        assert len(self.children) == 0 and len(self.parents) == 1
        current_node = self
        while current_node.youngest_parent() is not None and \
                current_node.youngest_parent().youngest_child() == current_node:
            current_node = current_node.youngest_parent()
        # Reaching the root without finding younger children -> self is node of
        # the leftmost's part of the graph.
        if current_node.youngest_parent() is None:
            # print "(1) Stopping right search operation on {0}".format(self)
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
            # print "(2) Found right merge partner {0} for {1}".format(current_node,
            #                                                          self)
            return current_node

        else:
            # print "(3) Stopping right search operation on {0}".format(self)
            return None

    @staticmethod
    def merge_new_children(children):
        """ The beta ranges corresponding to the given children should be
        connected. Do not input only first and last children of a recently
        analysed node but all the children that have been found."""
        if len(children) == 0:
            return [], []
        children.sort(key=lambda x: x.beta_max)
        left_candidate = children[0].find_left_merge_candidate()
        right_candidate = children[-1].find_right_merge_candidate()
        uncompleted_children = []
        children_for_stack = children[1:-1]  # Every element except first and last
        if len(children) == 1 and left_candidate is not None and \
                right_candidate is not None:
            # This is a special case since left and right candidate belong
            # to the same area which was previously intersected by the given
            # children[0] node. Here we have to be especially careful to restore
            # the correct relations, and the stack with which we go on
            # afterwards.

            # Assemble uncompleted children area first since we override
            # children
            if len(left_candidate.children) + len(right_candidate.children) > 0:
                if len(left_candidate.children) == 0:
                    uncompleted_children.append((left_candidate,
                                                 left_candidate.beta_min,
                                                 children[0].beta_max))
                elif len(right_candidate.children) == 0:
                    # Remark: The left candidate will be the surviving one with
                    # the correct relations, hence we need to append the left
                    # one here as well.
                    uncompleted_children.append((left_candidate,
                                                 children[0].beta_min,
                                                 right_candidate.beta_max))
                else:
                    # Remark: The left candidate will be the surviving one with
                    # the correct relations, hence we need to append the left
                    # one here as well.
                    # Case where both sourrounding nodes have already been
                    # processed.
                    uncompleted_children.append((left_candidate,
                                                 children[0].beta_min,
                                                 children[0].beta_max))
            print "Merging {0} with {1} and {2}".format(left_candidate,
                                                        children[0],
                                                        right_candidate)
            # Case we have searched for left and right candidates of single node
            # Case left_candidate + right_candidate + children node belong
            # to the same tiling element.
            left_candidate.alpha_max = right_candidate.alpha_max
            left_candidate.beta_max = right_candidate.beta_max
            left_candidate.parents += children[0].parents + right_candidate.parents
            left_candidate.uniquefy_parents()
            left_candidate.children += right_candidate.children
            left_candidate.uniquefy_children()
            # Fix parents of right candidate by replacing the child
            for parent in right_candidate.parents:
                parent[0].replace_child(right_candidate, left_candidate)
                parent[0].sort_children()
                parent[0].uniquefy_children()
            # Fix parents of children[0] by replacing the child
            for parent in children[0].parents:
                parent[0].replace_child(children[0], left_candidate)
                parent[0].sort_children()
                parent[0].uniquefy_children()
            # Fix children of right_candidate by replacing the parent
            for child in right_candidate.children:
                child[0].replace_parent(right_candidate, left_candidate)
                child[0].sort_parents()
                child[0].uniquefy_children()
        else:
            if left_candidate is not None:
                print "Merging {0} with {1}".format(left_candidate,
                                                    children[0])
                # Case left_candidate + right_candidate + children node belong
                # to the same tiling element.
                left_candidate.alpha_max = children[0].alpha_max
                left_candidate.beta_max = children[0].beta_max
                left_candidate.parents += children[0].parents
                left_candidate.sort_parents()
                left_candidate.uniquefy_parents()
                # Fix parents of children[0] by replacing the child
                for parent in children[0].parents:
                    parent[0].replace_child(children[0], left_candidate)
                    parent[0].sort_children()
                    parent[0].uniquefy_children()
                if len(left_candidate.children) > 0:
                    # In this case the left candidate is not in the 'to-process'
                    # stack anymore, hence we need to compute the left-over
                    # children directly.
                    uncompleted_children.append((left_candidate,
                                                 children[0].beta_min,
                                                 children[0].beta_max))
            else:
                children_for_stack.insert(0, children[0])
            if right_candidate is not None:
                print "Merging {0} with {1}".format(children[-1],
                                                    right_candidate)
                # Case left_candidate + right_candidate + children node belong
                # to the same tiling element.
                right_candidate.alpha_min = children[-1].alpha_min
                right_candidate.beta_min = children[-1].beta_min
                right_candidate.parents += children[-1].parents
                right_candidate.sort_parents()
                right_candidate.uniquefy_parents()
                # Fix parents of children[0] by replacing the child
                for parent in children[-1].parents:
                    parent[0].replace_child(children[-1], right_candidate)
                    parent[0].sort_children()
                    parent[0].uniquefy_children()
                if len(right_candidate.children) > 0:
                    # In this case the right candidate is not in the 'to-process'
                    # stack anymore, hence we need to compute the left-over
                    # children directly.
                    uncompleted_children.append((right_candidate,
                                                 children[-1].beta_min,
                                                 children[-1].beta_max))
            elif len(children) > 1:
                # If len(children) == 1 it is already on the stack if it
                # corresponding candidates were None
                children_for_stack.insert(len(children_for_stack), children[-1])
        return uncompleted_children, children_for_stack

    @staticmethod
    def base_region(beta_min, beta_max, A, y, svdU, svdS, options):
        return TilingElement(100.0, 100.0, beta_min, beta_max,
                             np.array([]).astype("uint32"),
                             np.array([]).astype("int32"), None, A, y, svdU,
                             svdS, options)

    def verify_tiling(self):
        """ Testing the integrity of the tree from this node on and below. A
        tree is considered valid if itself is valid and all its children are
        valid. A node is valid if the exists a distinct(!) child for all betas
        from which the node under consideration can be reached; or a node is
        valid if it has not a single node.

        Parameters
        -----------
        self : object of class TilingElement
            Root element of the (sub)-tree that shall be verified.

        Returns
        ------------
        True, if the (sub)-tree of this node is valid; false otherwise.
        """
        elements = self.bds_order()
        for element, layer in elements.iteritems():
            n_children = len(element.children)
            if n_children == 0:
                continue
            if element.children[0][1] != element.beta_min:
                print """Verification failed:
                         element.children[0][1]!=element.beta_min: {0}, {1}""".format(
                    element.children[0], element)
                return False
            ctr = 1
            while ctr < n_children:
                if element.children[ctr - 1][2] != element.children[ctr][1]:
                    print """Verification failed:
                             element.children[ctr-1][2]!=element.children[ctr][1]: {0}, {1}""".format(
                        element.children[ctr - 1], element.children[ctr])
                    return False
                ctr = ctr + 1
            if element.children[-1][2] != element.beta_max:
                print """Verification failed:
                         element.children[-1][2]!=element.beta_max: {0}, {1}""".format(
                    element.children[-1], element)
                return False
        print "Verification passed"
        return True

    def bds_order(self):
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

    def plot_tiling(self, n_disc=3):
        """ Plot the support tiling outgoing from the given node as a root. We
        use bds_order and the distance from the root_node to define some notion
        of a layered structure. """
        elements = self.bds_order()
        max_layer = max(elements.iteritems(), key=operator.itemgetter(1))[1]
        # Calculate elements in inverse relation ship, ie. layer to elements
        # instead of elements to layer (we will omit zeroth layer)
        reordered_dict = []
        for i in range(1, max_layer + 1):
            reordered_dict.append([e for e in elements.keys() if
                                   elements[e] == i])
            reordered_dict[-1].sort(key=lambda x: x.beta_min)
        # Make an adaptive beta discretisation that guarantees at least n_disc
        # points to represent the boundaries of each tiling element.
        beta_disc_coarse = []
        for element, layers in elements.iteritems():
            beta_disc_coarse.append(element.beta_min)
            beta_disc_coarse.append(element.beta_max)
        beta_disc_coarse = np.unique(np.array(beta_disc_coarse))
        beta_disc = np.zeros((len(beta_disc_coarse) - 1) * n_disc)
        for i in range(len(beta_disc_coarse) - 1):
            beta_disc[i * n_disc:(i + 1) * n_disc] = \
                np.linspace(beta_disc_coarse[i], beta_disc_coarse[i + 1],
                            n_disc)

        points = [np.zeros((beta_disc.shape[0], 2)) for _ in range(max_layer)]
        colors = [[] for _ in range(max_layer)]
        colorlist = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        for j, beta in enumerate(beta_disc):
            B_beta, y_beta = calc_B_y_beta(self.A, self.y, self.svdAAt_U,
                                           self.svdAAt_S, beta)
            for i, layer in enumerate(reordered_dict):
                idx, element = [(idx, e) for (idx, e) in enumerate(layer)
                                if e.beta_min <= beta and e.beta_max >= beta][0]
                prev_element = [pe[0] for pe in element.parents if pe[1] <= beta
                                and beta <= pe[2]][0]
                J = np.setxor1d(element.support, prev_element.support)
                alpha, sign = calc_all_cand(B_beta, y_beta,
                                            prev_element.support,
                                            prev_element.sign_pattern)
                points[i][j, 0] = beta
                points[i][j, 1] = alpha[J]
                colors[i].append(colorlist[idx % 6])
        plt.figure()
        ax = plt.gca()
        plt.title(r'Support tiling $u_{\alpha, \beta}$')
        for i in range(max_layer):
            for j in range(len(points[i]) - 1):
                plt.plot(points[i][j:j + 2, 0], points[i][j:j + 2, 1],
                         c=colors[i][j + 1], linewidth=2.0, alpha=0.5)
                if i == 0:
                    ax.fill_between(points[i][j:j + 2, 0],
                                    points[i][j:j + 2, 1],
                                    np.max(points[0][:, 1]),
                                    facecolor=colorlist[0],
                                    alpha=0.5, linewidth=0.0)
                elif i < max_layer:
                    ax.fill_between(points[i][j:j + 2, 0],
                                    points[i][j:j + 2, 1],
                                    points[i - 1][j:j + 2, 1],
                                    facecolor=colors[i - 1][j + 1],
                                    alpha=0.5, linewidth=0.0)
                    if i == max_layer - 1:
                        ax.fill_between(points[i][j:j + 2, 0], 0,
                                        points[i][j:j + 2, 1],
                                        facecolor=colors[i][j + 1],
                                        alpha=0.5, linewidth=0.0)
        plt.xlabel(r'$\beta$')
        plt.ylabel(r'$\alpha$')
        plt.show()

    def plot_graph(self, y_mode='layered'):
        vertices = self.bds_order()
        plt.figure()
        for element, layer in vertices.iteritems():
            # Skip root element
            if layer == 0:
                continue
            # Draw nodes
            if y_mode == 'layered':
                ycoord = -layer
            else:
                ycoord = 0.5 * (element.alpha_min + element.alpha_max)
            plt.scatter(0.5 * (element.beta_min + element.beta_max), ycoord,
                        s=50.0)
            # Draw edges
            for child in element.children:
                child_entry = [item for item in list(vertices.iteritems())
                               if item[0] == child[0]]
                xstart = 0.5 * (element.beta_min + element.beta_max)
                xend = 0.5 * (child[0].beta_min + child[0].beta_max) - \
                    0.5 * (element.beta_min + element.beta_max)
                if y_mode == 'layered':
                    ystart = -layer
                    yend = -child_entry[0][1] + layer
                else:
                    ystart = 0.5 * (element.alpha_min + element.alpha_max)
                    yend = 0.5 * (child[0].alpha_min + child[0].alpha_max) - \
                        0.5 * (element.alpha_min + element.alpha_max)
                plt.arrow(xstart, ystart, xend, yend, head_width=0.04,
                          head_length=0.075, fc="k", ec="k")  # , length_includes_head=True)
                plt.annotate("[{0},{1}]".format(child[1], child[2]),
                             xy=(0.5 * (2 * xstart + xend),
                                 0.5 * (2 * ystart + yend)),
                             xytext=(0.5 * (2 * xstart + xend),
                                     0.5 * (2 * ystart + yend)))
        plt.xlabel(r'$\beta$')
        plt.title('Support tiling graph (mode: {0})'.format(y_mode))
        plt.show()
