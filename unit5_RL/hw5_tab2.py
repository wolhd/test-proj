#homework5 tab2

import numpy as np

Q1_sC = np.zeros(5+1)
a = Q1_sC

a[1] = 0.3*((5/4)**-.5) + 0.7*(2**(1/3))

R1 = np.array([0, 5, 6, 7, 8, 9])
R2 = np.array([0, 2,   2,       2, 0,     0])
qcalc =lambda R1, R2: 0.3*(R1**-.5) + 0.7 *(R2**(1/3))
a[1:-2] = qcalc(R1[1:-2],R2[1:-2])

a[4] = 8**-.5
a[5] = 9**-.5


print(a)
