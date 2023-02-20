# -*- coding: utf-8 -*-
import numpy as np

# One data instance (one feature vector and label)
class Data:
    def __init__(self, x: list, y: int):
        self.x = np.array(x)
        self.y = y


# trainData array size: numData x xsize+1 (+1 for y)
#   thetaInit = ( [theta1,theta2], theta0)
def perceptron(trainData: np.array, xsize, T: int, thetaInit=None):
    theta = np.zeros(xsize)
    theta_0 = 0
    if thetaInit:
        theta = np.array(thetaInit[0])
        theta_0 = thetaInit[1]
    print("theta ", theta, theta_0)
    # num rows
    numData = trainData.shape[0]
    for t in range(T):
        print("start of run ", t)
        for i in range(numData):
            x = trainData[i,0:xsize]
            y = trainData[i,xsize]
            print("t %d i %d : x:%s y:%s theta:%s,%.2f" % 
                  (t, i, x, y, theta,theta_0))

            calc = y * (np.dot(theta, x) + theta_0)
            if calc <= 0 :
                print("negative ", calc, " x: ", x)
                theta = theta + y*x
                theta_0 = theta_0 + y
        print("end of run ", t)
    return (theta, theta_0)

#    [x1, x2,   y]
d = [
     [-4,   2,  1],
     [-2,   1,  1],
     [-1,  -1, -1],
     [ 2,   2, -1],
     [ 1,  -2, -1]
     ]
d = np.array(d)

perceptron(d, 2, 2)

# perceptron(d, 2,2, ([-1,1],-1.5))
