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

def plot_paths(true_path,est_path,obs_list,xEst,PEst):

    true_x = []
    true_y = []
    est_x = []
    est_y = []
    obs_x = []
    obs_y = []
    plt.cla()
    for i in range(len(true_path)):

        true_x.append(true_path[i][0][0])
        true_y.append(true_path[i][1][0])

        est_x.append(est_path[i][0][0])
        est_y.append(est_path[i][1][0])

        obs_x.append(obs_list[i][0][0])
        obs_y.append(obs_list[i][1][0])

    plt.plot(true_x,true_y,"-k")
    plt.plot(est_x,est_y,"-b")
    plt.plot(obs_x,obs_y,".g")
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
