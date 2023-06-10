import numpy as np
import matplotlib.pyplot as plt
import math

def theta2RMatrix(angle):

    ctheta = math.cos(angle)
    stheta = math.sin(angle)
    Rmat = np.array([[ctheta, -stheta, 0],
                     [stheta,  ctheta, 0],
                     [0, 0, 1]])
    
    return Rmat[0:2, 0:2]

def residual(a, b):
    """ compute residual (a-b) between measurements containing 
    [range, bearing]. Bearing is normalized to [-pi, pi)"""
    y = a - b
    y[1] = y[1] % (2 * np.pi)    # force in range [0, 2 pi)
    if y[1] > np.pi:             # move to [-pi, pi)
        y[1] -= 2 * np.pi
    return y

def plot_paths(results,xEst,PEst):

    plt.cla()
    plt.plot(results["true_path"][0, :].flatten(),results["true_path"][1, :].flatten(),"-k")
    plt.plot(results["est_path"][0, :].flatten(),results["est_path"][1, :].flatten(),"-b")
    plt.plot(results["observation"][0, :],results["observation"][1, :],".g")
    plot_covariance_ellipse(xEst, PEst)
        
    plt.title("EKF with GPS and Simple car model")
    plt.legend(["True path","Est. path"])
    plt.axis("equal")
    plt.grid(True)
    plt.pause(0.0001)
    # plt.show()

def plot_covariance_ellipse(xEst, PEst):  # pragma: no cover

    Pxy = PEst[0:2, 0:2]
    eigval, eigvec = np.linalg.eig(Pxy)

    if eigval[0] >= eigval[1]:
        bigind = 0
        smallind = 1
    else:
        bigind = 1
        smallind = 0

    t = np.arange(0, 2 * math.pi + 0.1, 0.1)
    a = math.sqrt(eigval[bigind])
    b = math.sqrt(eigval[smallind])
    x = [a * math.cos(it) for it in t]
    y = [b * math.sin(it) for it in t]
    angle = math.atan2(eigvec[1, bigind], eigvec[0, bigind])
    fx = theta2RMatrix(angle) @ (np.array([x, y]))
    px = np.array(fx[0, :] + xEst[0, 0]).flatten()
    py = np.array(fx[1, :] + xEst[1, 0]).flatten()
    plt.plot(px, py, "--r")
