import numpy as np
import math

class RangeBearing:

    def __init__(self):

        self.dim_z = 2

        self.noise = np.diag([0.5, 0.5])**2

        self.Q = np.diag([1.0,1.0])**2
    
    def solve(self,landmark,X):

        range = math.sqrt((landmark[0] - X[0])**2 + (landmark[1] - X[1])**2)
        bearing = math.atan2(landmark[1] - X[1],landmark[0] - X[0]) - X[2]

        Z_pred = np.array([[range,bearing]]).T

        J_z = self.get_jacobian(landmark,X)

        measurement = self.get_measurement(landmark)

        return measurement,Z_pred,J_z
    
    def get_measurement(self,Z):
        
        Z = Z + self.noise @ np.random.randn(self.dim_z,1)

        return Z
    
    def get_jacobian(self,lnd,x):

        dist = (lnd[0] - x[0])**2 + (lnd[1] - x[1])**2
        J_o = np.array([[-(lnd[0] - x[0])/math.sqrt(dist), -(lnd[1] - x[1])/math.sqrt(dist), 0],
                        [(lnd[1] - x[1])/dist, -(lnd[0] - x[0])/dist, -1]])
        
        return J_o
    
    def set_uncertainty(self,Q):
        
        assert type(Q) == np.ndarray, "Observation uncertainity not a matrix"
        assert len(Q.shape) == 2, "Measurement matrix is not a 2D matrix"
        assert Q.shape[0] == Q.shape[1], "Measurement matrix is not a square matrix"
        assert Q.shape[0] == self.dim_z, "Measurement matrix is not of same size as that of observation model"

        self.Q = Q