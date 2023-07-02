# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 13:28:07 2023

@author: myname
"""

#tab 3
import numpy as np

x = .95 * .75
print(x)

# theta^2*(ln(theta))^2/(ln(theta) + 1 - (ln(theta))^3)

theta = .62
n = 100
q = 1.645
v = theta**2*(np.log(theta))**2/(np.log(theta) + 1 - 
                                 (np.log(theta))**3)

B = theta + q*v**.5/n**.5
print('B', B)