import numpy as np
import math

class GPS:

    def __init__(self):

        self.dim_z = 2

        self.noise = np.diag([0.5, 0.5])**2

        self.Q = np.diag([1.0,1.0])**2
        self.C = np.array([[1, 0, 0],
                           [0, 1, 0]])
    
    def solve(self,X):

        assert type(X) == np.ndarray, "State vector is not a matrix"
        assert len(X.shape) == 2, "State vector is not a 2D matrix"
        assert X.shape[0] == self.C.shape[1], "State vector is not of same size as that of dimensions of OBSERVATION MODEL"

        Z_pred = self.C@X

        J_z = self.get_jacobian()

        measurement = self.get_measurement(Z_pred)

        return Z_pred,measurement,J_z
    
    def get_measurement(self,Z):

        Z = Z + self.noise@np.random.randn(self.dim_z,1)

        return Z
    
    def get_jacobian(self):

        J_o = np.array([[1, 0, 0],
                        [0, 1, 0]])
        
        return J_o
    
    def set_uncertainty(self,Q):
        
        assert type(Q) == np.ndarray, "Observation uncertainity not a matrix"
        assert len(Q.shape) == 2, "Measurement matrix is not a 2D matrix"
        assert Q.shape[0] == Q.shape[1], "Measurement matrix is not a square matrix"
        assert Q.shape[0] == self.dim_z, "Measurement matrix is not of same size as that of observation model"

        self.Q = Q