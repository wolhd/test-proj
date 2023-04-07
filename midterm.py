# midterm exam

import numpy as np

# 	(0,0)	(2,0)	(1,1)	(0,2) into Nx2 array
def parse_coordstr(coordstr):
    cstr = coordstr.split()
    cstr = [c.strip(' ()') for c in cstr]
    cstr = [c.split(',') for c in cstr]
    co = [[int(c[0]), int(c[1])] for c in cstr]
    return np.array(co)

# 1 6 4 2  into a Nx1 array
def parse_num_str(nstr):
    n = np.vstack(np.array( [ int(n) for n in nstr.split() ] ))
    return n

# probl 1


# labelstr = '-1	-1	-1	-1	-1	+1	+1	+1	+1	+1'
# labelstr = labelstr.split()
# labels = np.vstack(np.array( [ int(la) for la in labelstr] ))

# mistakes = '1	9	10	5	9	11	0	3	1	1'
# mistakes = mistakes.split()
# mistakes = np.vstack(np.array( [ int(m) for m in mistakes] ))

# coords = '	(0,0)	(2,0)	(3,0)	(0,2)	(2,2)	(5,1)	(5,2)	(2,4)	(4,4)	(5,5)'

# coords = parse_coordstr(coords)

# theta0 = (mistakes * labels).sum()
# print(theta0)
# theta = (coords * labels * mistakes).sum(axis=0)
# print(theta)


# # 1 (5)
# def hingeloss(coords, labels, theta, theta0):
#     loss = np.zeros((coords.shape[0],1))
#     for i in range(coords.shape[0]):
#         x = coords[i]
#         y = labels[i]
#         z = y * (np.dot(theta, x) + theta0)
#         if z >=1:
#             loss[i] = 0
#         else:
#             loss[i] = 1-z
#     return loss
# theta = np.array([1,1])
# theta0 = -5
# #1 (6)
# theta = theta/2
# theta0 = theta0/2
# loss = hingeloss(coords, labels, theta, theta0)
# sumloss = loss.sum()
# print(sumloss)

# probl 2

labels = '-1	-1	-1	-1	-1	+1	+1	+1	+1	+1'
labels = parse_num_str(labels)
mistakes = '1	65	11	31	72	30	0	21	4	15'
mistakes = parse_num_str(mistakes)
coords = '	(0,0)	(2,0)	(1,1)	(0,2)	(3,3)	(4,1)	(5,2)	(1,4)	(4,4)	(5,5)'
coords = parse_coordstr(coords)

theta0 = (mistakes * labels).sum()
phi = np.zeros((coords.shape[0],3))
phi[:,0] = coords[:,0]**2
phi[:,1] = np.sqrt(2)*coords[:,0]*coords[:,1]
phi[:,2] = coords[:,1]**2

# 2.2
zs = phi * labels * mistakes
theta = zs.sum(axis=0)

# 2.3
zclass = np.zeros((phi.shape[0],1))
for i in range(phi.shape[0]):
    z  = labels[i] * (np.dot(theta, phi[i,:]) + theta0)
    zclass[i] = z
