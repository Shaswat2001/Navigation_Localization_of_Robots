import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))

import numpy as np
import unittest
from Localization.ExtendedKalmanFilter import EKF
from MotionModel.simple_car.simple_car import SimpleCarModel
from ObservationModel.GPS.GPS import GPS

class EKFTest(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        motion_model = SimpleCarModel()
        obs_model = GPS()
        self.filter = EKF(motion_model,obs_model)

    def test_prediction_step(self):

        J_m = self.model.get_jacobian(np.array([[0,0,0]]).T)
        self.assertIsNone(np.testing.assert_allclose(J_m, np.array([[1, 0, 0],
                                                                    [0, 1, 0.1],
                                                                    [0, 0, 1]])), "Incorrect jacobian")

    def test_motion_model(self):

        x_new,_ = self.model.solve(np.array([[0,0,0]]).T)
        self.assertIsNone(np.testing.assert_allclose(x_new, np.array([[0.1,0,0.01]]).T), "Incorrect new state vector")

unittest.main()