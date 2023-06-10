import numpy as np
import math

class RangeBearing:

    def __init__(self):

        self.dim_z = 2

        self.noise = np.diag([0.3, 0.1])

        self.Q = np.diag([0.3,0.1])**2
    
    def solve(self,landmark,X):

        range = math.sqrt((landmark[0][0] - X[0][0])**2 + (landmark[1][0] - X[1][0])**2)
        bearing = math.atan2(landmark[1][0] - X[1][0],landmark[0][0] - X[0][0]) - X[2][0]

        Z_pred = np.array([[range,bearing]]).T

        J_z = self.get_jacobian(landmark,X)

        measurement = self.get_measurement(Z_pred)

        return measurement,Z_pred,J_z
    
    def get_measurement(self,Z):
        
        Z = Z + self.noise @ np.random.randn(self.dim_z,1)

        return Z
    
    def get_jacobian(self,lnd,x):

        dist = (lnd[0][0] - x[0][0])**2 + (lnd[1][0] - x[1][0])**2
        J_o = np.array([[-(lnd[0][0] - x[0][0])/math.sqrt(dist), -(lnd[1][0] - x[1][0])/math.sqrt(dist), 0],
                        [(lnd[1][0] - x[1][0])/dist, -(lnd[0][0] - x[0][0])/dist, -1]])
        
        return J_o
    
    def set_uncertainty(self,Q):
        
        assert type(Q) == np.ndarray, "Observation uncertainity not a matrix"
        assert len(Q.shape) == 2, "Measurement matrix is not a 2D matrix"
        assert Q.shape[0] == Q.shape[1], "Measurement matrix is not a square matrix"
        assert Q.shape[0] == self.dim_z, "Measurement matrix is not of same size as that of observation model"

        self.Q = Q