# coding: utf8
from itertools import groupby

import matplotlib.pyplot as plt
import numpy as np

from create_children_lars import (create_children_lars,
                                  lars_post_process_children)
from create_children_lasso import (create_children_lasso, lasso_children_merge,
                                   lasso_post_process_children)


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
                if (parent[1] <= beta_max and beta_min <= parent[2]): # Checks for overlap
                    new_index = np.setdiff1d(self.support, parent[0].support)[0]
                    additional_indices, boundary_parameters, used_signs = \
                        create_children_lasso(self.support, self.sign_pattern,
                                             np.maximum(beta_min, parent[1]), # beta_min
                                             np.minimum(beta_max, parent[2]), # beta_max
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
        return np.array_equal(tilingelement.support, self.support) and \
            np.array_equal(tilingelement.sign_pattern, self.sign_pattern)

    def can_be_merged_with(self, tilingelement):
        return self.shares_support_with(tilingelement) and \
                ((np.abs(self.alpha_max - tilingelement.alpha_min)/ \
                    self.alpha_max < 1e-4 and \
                  self.beta_max == tilingelement.beta_min) or \
                (np.abs(self.alpha_min - tilingelement.alpha_max)/ \
                    self.alpha_min < 1e-4 and \
                  self.beta_min == tilingelement.beta_max))

    def add_child(self, alpha_min, alpha_max, beta_min, beta_max, support,
                  signum):
        te = TilingElement(alpha_min, alpha_max, beta_min, beta_max, support,
                           signum, [[self, beta_min, beta_max]], self.A,
                           self.y, self.svdAAt_U, self.svdAAt_S, self.options)
        self.children.append([te, beta_min, beta_max])
        return te

    def sort_children(self):
        self.children.sort(key=lambda x: x[1])  # x[0] : beta_min

    def sort_parents(self):
        self.parents.sort(key=lambda x: x[1])  # x[0] : beta_min

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
                        child_to_replace, replacement))

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
                        parent_to_replace, replacement))

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
            import pdb
            pdb.set_trace()
            print ("Could not find next older neighbor of node {0} in " +
                   " children of node {1}.").format(child, self)
            return None

    def find_next_younger_neighbor(self, child):
        ctr = 0
        n_children = len(self.children)
        while ctr + 1 < n_children:
            if self.children[ctr][0] == child:
                return self.children[ctr + 1][0]
            ctr = ctr + 1
        else:
            import pdb
            pdb.set_trace()
            print ("Could not find next younger neighbor of node {0} in" +
                   " children of node {1}.").format(child, self)
            return None

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
        # Check: Parents of children should be empty
        # import pdb
        # pdb.set_trace()
        # for child in children:
        #     assert len(child.parents) == 0
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
                # print "Merging {0} with {1}".format(left_candidate,
                #                                     children[0])
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
                # print "Merging {0} with {1}".format(children[-1],
                #                                     right_candidate)
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
        tree is considered as being correct if
        (1) it has no children at all,
        (2) for all beta inside [beta_min, beta_max] there is a distinct child
            of this node.
        """
        n_children = len(self.children)
        if n_children == 0:
            return True
        else:
            # Check that shared boundary with first child starts at beta_min
            if self.children[0][1] != self.beta_min:
                print """Verification failed:
                         self.children[0][1]!=self.beta_min: {0}, {1}""".format(
                        self.children[0], self)
                return False
            ctr = 1
            while ctr < n_children:
                if self.children[ctr-1][2] != self.children[ctr][1]:
                    print """Verification failed:
                             self.children[ctr-1][2]!=self.children[ctr][1]: {0}, {1}""".format(
                            self.children[ctr-1], self.children[ctr])
                    return False
                ctr = ctr + 1
            if self.children[-1][2] != self.beta_max:
                print """Verification failed:
                         self.children[-1][2]!=self.beta_max: {0}, {1}""".format(
                        self.children[-1], self)
                return False
            return all([child[0].verify_tiling() for child in self.children])

    def bds_order(self):
        distance = 0
        elements_in_bds_order = {self : distance}
        element_queue = [self]
        while len(element_queue) > 0:
            element = element_queue.pop(0)
            distance += 1
            for child in element.children:
                if child[0] not in elements_in_bds_order:
                    elements_in_bds_order[child[0]] =  distance
                    element_queue.append(child[0])
        return elements_in_bds_order

    def plot_graph(self, y_mode = 'layered'):
        vertices = self.bds_order()
        plt.figure()
        for  element, layer in vertices.iteritems():
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
                          head_length=0.075, fc="k", ec="k") #, length_includes_head=True)
                plt.annotate("[{0},{1}]".format(child[1], child[2]),
                             xy=(0.5 * (2 * xstart + xend),
                                 0.5 * (2 * ystart + yend)),
                             xytext=(0.5 * (2 * xstart + xend),
                                     0.5 * (2 * ystart + yend)))
        plt.xlabel('beta')
        plt.title('Support tiling graph (mode: {0})'.format(y_mode))
        plt.show()
