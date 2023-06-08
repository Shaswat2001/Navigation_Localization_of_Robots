import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))


import numpy as np
from MotionModel.simple_car.simple_car import SimpleCarModel


if __name__ == "__main__":

    X = None
    SIM_TIME = 50

