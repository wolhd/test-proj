# Unit 5 reinforcement learning
# Lecture 17
# tab 7

# more complicated transitions 

import numpy as np

gamma = 0.75
Nstates = 4
Nactions = 2

states = np.arange(1,Nstates+1)
actions = np.arange(1, Nactions+1)

# T (s,a,sP),  T(0 0 0) is unused, so 5 states need length 6
T = np.zeros((Nstates+1, Nactions+1, Nstates+1))



# action 1: stay
# T(s, s, a=1, 0.5) # stay successful 1/2
# T(s=lfEnd,s=lfEnd+1, a=1, 0.5)
# T(s=rtEnd,s=rtEnd-1, a=1, 0.5)
# T(s=!rt_or_LfEnd, s=s+1, a=1, .25)
# T(s=!rt_or_LfEnd, s=s-1, a=1, .25)

# action 2: move left
# T(s=!lfEnd, s, a=2, 2/3) # fail to move
# T(s=!lfEnd, s-1, a=2, 1/3) # move success
# T(lfEnd, lfEnd, a=2, 0.5)
# T(lfEnd, lfEnd+1, a=2, 0.5)

# action 3: move right
# T(s=!rtEnd, s, a=3, 2/3) # fail to move
# T(s=!rtEnd, s+1, a=3, 1/3) # move success
# T(s=rtEnd, s=rtEnd, a=3, 0.5)
# T(s=rtEnd, rtEnd-1, a=3, 0.5)


def setTold():
    lfEnd = states[0]
    rtEnd = states[-1]
    inner = states[1:-1]
    notLf = states[1:]
    notRt = states[:-1]
    
    a = 1  # action: stay
    for s in states:
        T[s,a,s] = 0.5  # stay successful
    T[lfEnd, a, lfEnd+1] = 0.5 # left end moves
    T[rtEnd, a, rtEnd-1] = 0.5 # right end moves
    for s in inner:
        T[s,a,s+1] = 0.25 # inner moves right
        T[s,a,s-1] = 0.25 # inner moves left
        
    a = 2 # move left
    for s in notLf:
        T[s, a, s] = 2/3 # fail to move
        T[s, a, s-1] = 1/3 # move successful
    T[lfEnd, a, lfEnd] = 0.5 # left end no move
    T[lfEnd, a, lfEnd+1] = 0.5 # left end move successful
    
    a = 3 # move right
    for s in notRt:
        T[s, a, s] = 2/3 # fail to move
        T[s, a, s+1] = 1/3 # move succ
    T[rtEnd, a, rtEnd] = 0.5 # no move
    T[rtEnd, a, rtEnd-1] = 0.5 # move succ
    
    
# a=1 is move up
# a=2 is move down    
def setT():
    T[1,1,2] = 1
    T[2,1,3] = 1
    T[3,1,4] = 1
    T[4,2,3] = 1
    T[3,2,2] = 1
    T[2,2,1] = 1

    
setT()


# V (steps, states), state 0 is unused so states len is +1
# V[0]
V = np.zeros((1,Nstates+1)) 
step = 0

def Rold(s):
    return 1 if s==5 else 0

Rmat = np.zeros((Nstates+1, Nactions+1, Nstates+1))
def setRmat():
    Rmat[1,1,2] = 1
    Rmat[2,1,3] = 1
    Rmat[3,1,4] = 10
    Rmat[4,2,3] = 10
    Rmat[3,2,2] = 1
    Rmat[2,2,1] = 1

setRmat()
    
def R(s,a,sP):
    return Rmat[s,a,sP]

# calc for one transistion
def transis(s, a, sP, step):
  val = T[s,a,sP] * (R(s,a,sP) + gamma * V[step-1, sP])
  return val


def transis_sum_sP(s, a, step):
    sum = 0
    for sP in states:
        val = transis(s, a, sP, step)
        sum += val
    return sum

def Qmax_a(s, step):
    # if s == 5 and step > 0:
    #     return 1
    max_val = 0
    max_a = 0
    for a in actions:
        val = transis_sum_sP(s, a, step)
        if val > max_val:
            max_val = val 
            max_a = a
    return max_val, max_a

lastStep = 0
maxStep = 80
Amax = np.zeros(Nstates+1)
while True:
    step += 1
    V = np.append(V, np.zeros((1,Nstates+1)), axis=0)
    for state in states:
        val, a = Qmax_a(state, step)
        V[step, state] = val
        Amax[state] = a
    # if (V[step] == V[step-1]).all():
    #     lastStep = step-1
    #     break
    print('>>> a max', Amax)
    print('>>> V ',step, V[step])
    if step == maxStep:
        lastStep = step
        break
    
print(f" finished step {lastStep}, V {V[lastStep]}") 

       
