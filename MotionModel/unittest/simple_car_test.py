import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))

import unittest
import numpy as np
from simple_car.simple_car import SimpleCarModel

class SimpleCarModelTest(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.model = SimpleCarModel()

    def test_motion_model(self):

        x_new = self.model.motion_model(np.array([[0,0,0]]).T)
        self.assertIsNone(np.testing.assert_allclose(x_new, np.array([[0.1,0,0.01]]).T), "Incorrect new state vector")

unittest.main()
    

