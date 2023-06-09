import numpy as np
import math

class SimpleCarModel:

    def __init__(self):

        self.dim_state = 3 # [x, y, theta]
        self.dim_control = 2 # [v, omega]
        self.DT = 0.1

        self.R = np.diag([0.1, 0.1, np.deg2rad(1.0)])**2 # state variance in [x, y, theta]
        self.u = np.array([[1,0.1]]).T # control input [v, omega]

        self.A = np.eye(self.dim_state)

    def solve(self,X):

        assert type(X) == np.ndarray, "State vector is not a matrix"
        assert len(X.shape) == 2, "State vector is not a 2D matrix"
        assert X.shape[0] == self.dim_state, "State vector is not of same size as that of dimensions of SIMPLE CAR MODEL"

        B = np.array([[self.DT*math.cos(X[2][0]), 0],
                      [self.DT*math.sin(X[2][0]), 0],
                      [0, self.DT]])
        
        X_new = self.A @ X + B @ self.u

        J_m = self.get_jacobian(X)

        return X_new,J_m
    
    def get_jacobian(self,X):

        J_m = np.array([[1, 0, -self.DT*self.u[0][0]*math.sin(X[2])],
                        [0, 1,  self.DT*self.u[0][0]*math.cos(X[2])],
                        [0, 0, 1]])
        
        return J_m
    
    def set_controls(self,U):

        assert type(U) == np.ndarray, "Control vector is not a matrix"
        assert len(U.shape) == 2, "Control vector is not a 2D matrix"
        assert U.shape[0] == self.dim_control, "control vector is not of same size as that of dimensions of MODEL CONTROL"

        self.u = U

    def set_uncertainty(self,R):
        
        assert type(R) == np.ndarray, "State transition uncertainity not a matrix"
        assert len(R.shape) == 2, "State transition matrix is not a 2D matrix"
        assert R.shape[0] == R.shape[1], "State transition matrix is not a square matrix"
        assert R.shape[0] == self.dim_state, "State transition matrix is not of same size as the state vector"

        self.R = R

