import numpy as np

# stats course

# homework 9 bayesian

# tab 2

# (b)
n = 6
sumx = 4

def postfunc(lamb, sumx, n, prior):
    post = lamb**sumx * (1-lamb)**(n-sumx)*prior
    return post

postArr = np.array([])

lamb = 0.2
prior = 1/5
post = postfunc(lamb, sumx, n, prior)
print('lamb', lamb, 'post', post)
postArr = np.append(postArr, post)

lamb = 0.4
prior = 2/5
post = postfunc(lamb, sumx, n, prior)
print('lamb', lamb, 'post', post)
postArr = np.append(postArr, post)

lamb = 0.6
prior = 1/5
post = postfunc(lamb, sumx, n, prior)
print('lamb', lamb, 'post', post)
postArr = np.append(postArr, post)

lamb = 0.8
prior = 1/5
post = postfunc(lamb, sumx, n, prior)
print('lamb', lamb, 'post', post)
postArr = np.append(postArr, post)

postNormArr = postArr/postArr.sum()
print('postNormArr', postNormArr)
lambArr = np.array([0.2,0.4,.6,.8])
postmean = (lambArr*postNormArr).sum()
print('mean', postmean)