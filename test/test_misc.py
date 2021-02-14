import unittest

import numpy as np

from src import misc

class TestMisc(unittest.TestCase):
    def setUp(self):
        self.arr = np.array([[1, 2, 3, 4], [1, 5, 6, 7]])
        self.soft_max_res = np.array([[0.0321, 0.0871, 0.2369, 0.6439], [0.0016, 0.0899, 0.2443, 0.6641]])
        self.torrance = 1e-5

    def test_softmax(self):
        res = misc.softmax(self.arr)
        diff = np.sum((res - self.soft_max_res) ** 2)
        self.assertLess(diff, self.torrance)
