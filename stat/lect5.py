#Unit 2 lect 5

# tab 9

q_alpha2 = 1.65
n =64
xbar_n = 7.8

I_Low = 1/xbar_n * 1 / (1 + (q_alpha2/n**.5))
I_Up = 1/xbar_n * 1 / (1 - (q_alpha2/n**.5))
print(I_Low, I_Up, I_Up-I_Low)

I_Low = 1/xbar_n *  (1 - (q_alpha2/n**.5))
I_Up = 1/xbar_n *  (1 + (q_alpha2/n**.5))
print(I_Low, I_Up, I_Up-I_Low)
