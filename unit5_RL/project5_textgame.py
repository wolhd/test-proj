import numpy as np

# optimal episodic reward

# reward 2 rooms away
rew2Rms = -.01 * (.5)**0 - .01 * (.5) ** 1 + 1 * (.5) ** 2

# reward 1 room away
rew1Rm = -.01 + .5

# reward at room
rew0Rm = 1

# sum start at 4 rooms
rewSum = rew2Rms + 2 * rew1Rm + rew0Rm

# expectation
rewExpe = rewSum / 4
print(rewExpe)