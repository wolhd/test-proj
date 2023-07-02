import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# quadratic equ solver
# -b +- sqrt(b^2 - 4ac) / (2a)

# ax^2 + bx + c = 0

def quadraticSolve(a,b,c):
    sqrtTerm = (b**2 - 4*a*c)**.5
    x1 = (-1*b + sqrtTerm) / (2*a)
    x2 = (-1*b - sqrtTerm) / (2*a)
    return x1, x2

a = 1
b = -4 
c = 4


x1, x2 = quadraticSolve(a, b, c)

print(x1, x2)

x = np.arange(-10, 10, step=0.5)
y = a*x**2 + b * x + c
# plt.plot(x,y, 'x')
# plt.grid()

#-----
alpha = .05
alpha_d2 = alpha / 2
# q_alpha/2 is the norm(0,1) x for cdf(1-alpha/2), cdf(.975)
q_alpha_d2 = norm.ppf(1-alpha_d2)   # 1.96

n = 100
Rbar_n = .645

A = 1+ q_alpha_d2**2 / n
B = -1 * (2*Rbar_n + q_alpha_d2**2 / n)
C = Rbar_n ** 2
print(A,B,C)

p1,p2 = quadraticSolve(A, B, C)
print(p1, p2) 
x = np.arange(.3,.8,step=.02)
y = A*x**2 + B*x + C
plt.plot(x,y)
plt.grid()

