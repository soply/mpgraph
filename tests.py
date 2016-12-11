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
                                A, y, svdU, svdS, options = options)
    stack.append(root_element)
    while len(stack) > 0:
        current_element = stack.pop(0)
        new_children = current_element.find_children()
        stack.extend(new_children)
if __name__ == "__main__":
    testcase = "TC1"
    test_graph_operations(testcase)
