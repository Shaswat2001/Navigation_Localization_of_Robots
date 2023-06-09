import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

import numpy as np
import unittest
from Localization.UnscentedKalmanFilter import UKF
from MotionModel.SimpleCar import SimpleCarModel
from ObservationModel.GPS import GPS

class EKFTest(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        motion_model = SimpleCarModel()
        obs_model = GPS()
        self.filter = UKF(motion_model,obs_model)

    def test_sigma_points(self):

        points = self.filter.X
        self.assertIsNone(np.testing.assert_allclose(points, np.array([[ 0, 0, 0],
                                                                        [ 0.52820451, 0, 0],
                                                                        [ 0, 0.52820451, 0],
                                                                        [ 0, 0,  0.52820451],
                                                                        [-0.52820451, 0, 0],
                                                                        [ 0, -0.52820451, 0],
                                                                        [ 0, 0, -0.52820451]])), "Incorrect sigma points")


    def test_motion_model_mean(self):

        points_sum = self.filter.calculate_weight_sum(self.filter.X)
        self.assertIsNone(np.testing.assert_allclose(points_sum, np.array([[ 0, 0, 0]]).T), "Incorrect sigma points")
    
    def test_motion_model_covariance(self):

        points_sum = self.filter.calculate_weight_sum(self.filter.X)
        variance = self.filter.calculate_estimated_variance(self.filter.X,points_sum)
        self.assertIsNone(np.testing.assert_allclose(variance, np.array([[1, 0, 0],
                                                                         [0, 1, 0],
                                                                         [0, 0, 1]])), "Incorrect sigma points")

    # def test_prediction_step(self):

    #     _,P_prior = self.filter.prediction_step()
        
    #     self.assertIsNone(np.testing.assert_allclose(P_prior, np.array([[1.01, 0, 0],
    #                                                                 [0, 1.02, 0.1],
    #                                                                 [0, 0.1, 1.0003046]])), "Incorrect prediction covariance")

    # def test_kalman_filter(self):

    #     Pp = np.eye(3)
    #     H = np.array([[1, 0, 0],
    #                   [0, 1, 0]])
    #     Q = np.eye(2)

    #     kalman = self.filter.calculate_kalman_gain(Pp,H,Q)

    #     self.assertIsNone(np.testing.assert_allclose(kalman, np.array([[2, 0],
    #                                                                    [0, 2],
    #                                                                    [0, 0]])), "Incorrect Kalman gain")

unittest.main()