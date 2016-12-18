#coding: utf8
import numpy as np

import test_utils.test_graph_operations as tgo
from tilingElement import TilingElement


def test_graph_operations(testcase):
    region_iterator = tgo.create_test_case_iterator(testcase)
    A, y, svdU, svdS = tgo.create_dummy_variables()
    base_region = region_iterator.next()[0]
    options = {
        "mode" : "TEST",
        "test_iterator": region_iterator
    }
    stack = []
    root_element = TilingElement(base_region[0][1], base_region[1][1],
                                base_region[2], base_region[3], None,
                                A, y, svdU, svdS, 0, options = options)
    n_element = 1
    stack.append(root_element)
    while len(stack) > 0:
        current_element = stack.pop(0)
        try:
            children = current_element.find_children(n_element)
            print children
            uncompleted_children, children_for_stack = \
                                    TilingElement.merge_new_children(children)
            n_element += len(children_for_stack)
            stack.extend(children_for_stack)
            while len(uncompleted_children) > 0:
                uncomp_child, beta_min, beta_max = uncompleted_children.pop(0)
                children = uncomp_child.find_children(n_element, beta_min,
                                                        beta_max)
                tmp_uncomp_children, children_for_stack = \
                                    TilingElement.merge_new_children(children)
                uncompleted_children.extend(tmp_uncomp_children)
                n_element += len(children_for_stack)
                stack.extend(children_for_stack)
        except StopIteration:
            root_element.plot_graph()
            import pdb
            pdb.set_trace()
if __name__ == "__main__":
    testcase = "TC4"
    test_graph_operations(testcase)
