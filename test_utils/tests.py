#coding: utf8
import os.path
import sys

import numpy as np

import testcases_graph_operations as tgo

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
print os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)

from tilingElement import TilingElement


def test_graph_operations(testcase):
    region_iterator = tgo.create_test_case_iterator(testcase)
    A, y, svdU, svdS = tgo.create_dummy_variables()
    base_region = region_iterator.next()[0]
    options = {
        "verbose" : 2,
        "mode" : "TEST",
        "test_iterator": region_iterator,
        "env_minimiser": "scipy_brentq"
    }
    stack = []
    root_element = TilingElement(base_region[0][0], base_region[1][0],
                                base_region[0][1], base_region[1][1],
                                base_region[2], base_region[3], None,
                                A, y, svdU, svdS, options = options)
    stack.append(root_element)
    while len(stack) > 0:
        current_element = stack.pop(0)
        try:
            children = current_element.find_children()
            uncompleted_children, children_for_stack = \
                                    TilingElement.merge_new_children(children)
            stack.extend(children_for_stack)
            while len(uncompleted_children) > 0:
                uncomp_child, beta_min, beta_max = uncompleted_children.pop(0)
                children = uncomp_child.find_children(beta_min, beta_max)
                tmp_uncomp_children, children_for_stack = \
                                    TilingElement.merge_new_children(children)
                uncompleted_children.extend(tmp_uncomp_children)
                stack.extend(children_for_stack)
        except StopIteration:
            import pdb
            pdb.set_trace()

if __name__ == "__main__":
    testcase = "TC6"
    test_graph_operations(testcase)
