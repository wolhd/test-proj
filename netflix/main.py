import numpy as np
import kmeans
import common
import naive_em
import em

X = np.loadtxt("toy_data.txt")

# TODO: Your code here

Ks = [1,2,3,4]
seeds = [0,1,2,3,4]

K = Ks[0]
seed = seeds[0]
lowest_costs = []
for K in Ks:
    print(f"run K {K} =====================================")
    low_cost = None
    for seed in seeds:
        print(f"seed {seed} -------------------------- ")
        mixture, post = common.init(X, K, seed)
        print(f" initial mixture {mixture}, post {post[:4,:]}")
        
        ret_mixture, ret_post, cost = kmeans.run(X, mixture, post)
        
        print(f"result mixture {ret_mixture}, post {ret_post[:4,:]}, cost {cost}")
        
        if low_cost == None:
            low_cost = cost
        else:
            if cost < low_cost:
                low_cost = cost


        common.plot(X, ret_mixture, ret_post, f"cost {cost}")
    lowest_costs.append(low_cost)
    
print(f"lowest costs {lowest_costs}")
