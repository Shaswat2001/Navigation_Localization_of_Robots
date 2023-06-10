import numpy as np
import math

class SimpleCarModel:

    def __init__(self):
        
        self.dim_state = 3 # [x, y, theta]
        self.dim_control = 2 # [v, omega]
        self.DT = 0.1

        self.noise = np.diag([1.0, np.deg2rad(3.0)])**2 # control error given to velocity commands
        self.M = np.diag([0.1, np.deg2rad(1.0)])**2 # state variance in [x, y, theta]
        self.u = np.array([[1.1,0.01]]).T # control input [v, omega]

        self.A = np.eye(self.dim_state)

    def solve(self,X,method="prob"):

        assert type(X) == np.ndarray, "State vector is not a matrix"
        assert len(X.shape) == 2, "State vector is not a 2D matrix"
        assert X.shape[0] == self.dim_state, "State vector is not of same size as that of dimensions of SIMPLE CAR MODEL"

        omg = self.u[1][0]

        B = np.array([[-math.sin(X[2][0])/omg + math.sin(X[2][0] + self.DT*omg)/omg, 0],
                      [ math.cos(X[2][0])/omg - math.cos(X[2][0] + self.DT*omg)/omg, 0],
                      [0, self.DT]])
        
        if method == "prob":
            u = self.u + self.noise @ np.random.randn(2,1)
        else:
            u = self.u

        X_new = self.A @ X + B @ u

        J_m = self.get_jacobian(X,u)

        V = self.get_model_covariance(X,u)

        Rt = V @ self.M @ V.T

        return X_new,J_m,Rt
    
    def get_jacobian(self,X,u):

        v = u[0][0]
        omg = u[1][0]
        R = v/omg
        J_m = np.array([[1, 0, -R*math.cos(X[2][0]) + R*math.cos(X[2][0] + self.DT*omg)],
                        [0, 1, -R*math.sin(X[2][0]) + R*math.sin(X[2][0] + self.DT*omg)],
                        [0, 0, 1]])
        
        return J_m
    
    def get_model_covariance(self,X,u):
        
        v = u[0][0]
        omg = u[1][0]
        V = np.array([[(-math.sin(X[2][0]) + math.sin(X[2][0] + omg*self.DT))/omg,v*(math.sin(X[2][0]) - math.sin(X[2][0] + omg*self.DT))/omg**2 + v*math.cos(X[2][0] + omg*self.DT)*self.DT/omg],
                      [(math.cos(X[2][0]) - math.cos(X[2][0] + omg*self.DT))/omg,v*(-math.cos(X[2][0]) + math.cos(X[2][0] + omg*self.DT))/omg**2 + v*math.sin(X[2][0] + omg*self.DT)*self.DT/omg],
                      [0,self.DT]])
        
        return V

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

