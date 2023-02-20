# -*- coding: utf-8 -*-
import numpy as np

# One data instance (one feature vector and label)
class Data:
    def __init__(self, x: list, y: int):
        self.x = np.array(x)
        self.y = y


# perceptron algorithm for theta through origin (no theta_0)
# trainData array size: numData x xsize+1 (+1 for y)
def perceptronOrigin(trainData: np.array, xsize, T: int):
    theta = np.zeros(xsize)
    # num rows
    numData = trainData.shape[0]
    for t in range(T):
        for i in range(numData):
            print("t %d i %d : %s" % (t, i, theta))
            x = trainData[i,0:xsize]
            y = trainData[i,xsize]
            calc = y * np.dot(theta, x)
            if calc <= 0 :
                print("negative ", calc)
                theta = theta + y*x
    return theta

#    [x1, x2,   y]
d = [
     [-1,  -1,  1],
     [ 1,   0, -1],
     [-1, 1.5,  1] ]
d = np.array(d)
# perceptronOrigin(d, 2, 4)
# print("----------")
# d2 = np.roll(d, -1, axis=0)
# perceptronOrigin(d2, 2, 4)

# 1(c)
dc = d
#dc[2,0:2] = [-1, 10]
dc[2,0:2] = [-1/10.05, 10/10.05]

print(dc)
perceptronOrigin(dc, 2, 7)

# dc2 = np.roll(dc, -1, axis=0)
# print(dc2)
# perceptronOrigin(dc2, 2, 3)
