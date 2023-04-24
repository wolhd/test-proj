import numpy as np
import kmeans
import common
import naive_em
import em

X = np.loadtxt("toy_data.txt")

# try EM with toy data
# K = 3
# seed = 0

Ks = [1,2,3,4]
seeds = [0,1,2,3,4]
logliks = []

# for K in Ks:
#     maxll = -1e10
#     maxseed = -1
#     for seed in seeds:
    
#         gm, softCounts = common.init(X,K,seed)
#         # softCounts (N,K)
#         mix, cnts, loglik = naive_em.run(X, gm, softCounts)
#         if loglik > maxll:
#             maxll = loglik
#             maxseed = seed
#     print(f"K {K} seed {maxseed}, loglik={maxll}")




def bicTest():
    gm, scnts = common.init(X,14,99)
    bic = common.bic(X,gm,-1000)
    print(f"bic {bic}")
    
bicTest()

# TODO: Your code here
def kmeansTest(X):
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




