import numpy as np

from create_children_LARS import create_children_LARS, post_process_children


class TilingElement(object):

    def __init__(self, beta_min, beta_max, support, sign_pattern, parents,
                 A, y, svdAAt_U, svdAAt_S, children=None, options=None):
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.support = support
        self.sign_pattern = sign_pattern
        self.parents = parents
        if children is not None:
            self.children = children
        else:
            self.children = []
        self.mode = mode
        # References to global data matrices and data
        self.A = A
        self.y = y
        self.svdAAt_U = svdAAt_U
        self.svdAAt_S = svdAAt_S

    def __str__(self):
        return """Tiling element (Support: {0}, Sign pattern: {1},
                Beta range: {2})""".format(self.support, self.sign_pattern,
                                        [self.beta_min, self.beta_max])

    def create_children(self, beta_min=None, beta_max=None):
        if beta_min is None:
            beta_min = self.beta_min
        if beta_max is None:
            beta_max = self.beta_max
        if self.options["mode"] == "LARS":
            additional_indices, boundary_parameters, used_signs = \
                create_children_LARS(self.support, self.signum, beta_min,
                                     beta_max,
                                     self.options["env_minimiser"],
                                     self.svdAAt_U, self.svdAAt_S, self.A,
                                     self.y)
            children = post_process_children(additional_indices,
                                             boundary_parameters,
                                             used_signs, self.support,
                                             self.old_sign)
        else:
            raise NotImplementedError("Mode not implemented.")
        for child in reversed(children):
            self.children.append(TilingElement(
                child[0][1], child[1][1], child[2], child[3], [self],
                self.A, self.y, self.svdAAt_U, self.svdAAt_S, self.options)
        # Enforce that children are ordered correctly
        self.children.sort(key=lambda x: x.beta_min)
        new_childless_tilingelements=self.children
        new_childless_tilingelements_left=self.oldest_child.merge_left()
        new_childless_tilingelements_right=self.youngest_child.merge_right()
        return new_childless_tilingelements + \
                new_childless_tilingelements_left + \
                new_childless_tilingelements_right


    def compare(self, tilingelement):
        return np.array_equal(tilingelement.support, self.support) and \
            np.array_equal(tilingelement.sign_pattern, self.sign_pattern)

    def replace_child(self, child_to_replace, replacement):
        ctr=0:
        while ctr < len(self.children):
            if self.children[ctr] != child_to_replace:
                ctr += 1
            else:
                self.children[ctr]=replacement
                return 1
        else:
            print """Could not find child {0} in the children of child {1} and
                thus could not replace it with child {2}""".format(self,
                child_to_replace, replacement)

    def oldest_child(self):
        if len(self.children) > 0:
            return self.children[0]
        else:
            print """Tiling element: {0}\nNo children but asking for the
                oldest child. """.format(self)
            return None

    def youngest_child(self):
        if len(self.children) > 0:
            return self.children[-1]
        else:
            print """Tiling element: {0}\nNo children but asking for the
                youngest child. """.format(self)
            return None

    def oldest_parent(self):
        if len(self.parents) > 0:
            return self.parents[0]
        else:
            print """Tiling element: {0}\nNo parent but asking for the oldest
                parent. """.format(self)
            return None

    def youngest_parent(self):
        if len(self.parents) > 0:
            return self.parents[-1]
        else:
            print """Tiling element: {0}\nNo parent but asking for the oldest
                parent. """.format(self)
            return None


    def find_next_older_neighbor(self, child):
        ctr=1:
        n_children=len(self.children)
        while ctr < n_children:
            if self.children[ctr] == child:
                return self.children[ctr - 1]
        else:
            print """Could not find next older neighbor of node {0} in children
                of node {1}.""".format(child, self)
            return None

    def find_next_younger_neighbor(self, child):
        ctr=0:
        n_children=len(self.children)
        while ctr + 1 < n_children:
            if self.children[ctr + 1] == child:
                return self.children[ctr + 1]
            ctr = ctr + 1
        else:
            print """Could not find next younger neighbor of node {0} in
                children of node {1}.""".format(child, self)
            return None

    def merge_right(self):
        current_node=self
        while current_node.youngest_parent() is not None and \
                current_node.youngest_parent.oldest_child() == current_node:
            current_node=current_node.youngest_parent()
        if current_node.youngest_parent() is None:
            # In case we reached the root without having found a candidate for
            # merging the node
            return 0, []
        # Get oldest parent
        parent=current_node.youngest_parent()
        # Get next older sibling
        current_node=parent.find_next_younger_neighbor(current_node)
        # Find matching support and sign pattern
        while not current_node.compare(self) and \
                    len(current_node.children) > 0:
            current_node=current_node.oldest_child()
        if current_node is not None and current_node.compare(self):
            # Found a node to merge with
            # Fix beta max and parents! Note that the self node should not have
            # children yet.
            print """Merging nodes {0} and {1}""".format(current_node, self)
            current_node.beta_min=self.beta_min
            for parent in self.parents:
                parent.replace_child(self, current_node)
            if len(current_node.children) > 0:
                new_childless_tilingelements=current_node.create_children(
                                                beta_min=self.beta_min,
                                                beta_max=self.beta_max)
                return 1, new_childless_tilingelements
            else:
                return 1, []
        else:
            return 0, []

    def merge_left(self):
        current_node=self
        while current_node.oldest_parent() is not None and \
                current_node.oldest_parent.youngest_child() == current_node:
            current_node=current_node.oldest_parent()
        if current_node.oldest_parent() is None:
            # In case we reached the root without having found a candidate for
            # merging the node
            return 0, []
        # Get oldest parent
        parent=current_node.oldest_parent()
        # Get next older sibling
        current_node=parent.find_next_older_neighbor(current_node)
        # Find matching support and sign pattern
        while not current_node.compare(self) and \
                    len(current_node.children) > 0:
            current_node=current_node.youngest_child()
        if current_node is not None and current_node.compare(self):
            # Found a node to merge with
            # Fix beta max and parents! Note that the self node should not have
            # children yet.
            print """Merging nodes {0} and {1}""".format(current_node, self)
            current_node.beta_max=self.beta_max
            for parent in self.parents:
                parent.replace_child(self, current_node)
            if len(current_node.children) > 0:
                new_childless_tilingelements=current_node.create_children(
                                                beta_min=self.beta_min,
                                                beta_max=self.beta_max)
                return 1, new_childless_tilingelements
        else:
            return 0, []

    def verify_tiling(self):
        """ Testing the integrity of the tree from this node on and below. A
        tree is considered as being correct if
        (1) it has no children at all,
        (2) the children's beta_min's and beta_max's of the children yield a
            discretisation of this node's [beta_min, beta_max] range without any
            holes and they span the same range [beta_min, beta_max] in total.
        """
        ctr = 0
        n_children = len(self.children)
        if n_children == 0:
            return True
        elif self.beta_min != self.oldest_child().beta_min:
            return False
        elif self.beta_max != self.youngest_child().beta_max:
            return False

        else:
            while ctr + 1 < n_children:
                if self.children[ctr].beta_max == self.children[ctr+1].beta_min:
                    ctr = ctr + 1
                else:
                    return False
            # A tree is only correct if the same holds for all its children
            return all([child.verify_tiling() for child in self.children])
