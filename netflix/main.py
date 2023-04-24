import numpy as np
import kmeans
import common
import naive_em

X = np.loadtxt("toy_data.txt")

# try EM with toy data
# K = 3
# seed = 0


## Compare KMeans and EM

def emTest(X, useBic=False):
    Ks = [1,2,3,4]
    seeds = [0,1,2,3,4]
    logliks = np.zeros(np.array(Ks).max()+1)
    bestPlotArgs = dict()
    
    for K in Ks:
        maxll = -1e10
        maxseed = -1
        for seed in seeds:
        
            gm, softCounts = common.init(X,K,seed)
            # softCounts (N,K)
            mix, cnts, loglik = naive_em.run(X, gm, softCounts)
                    
            #common.plot(X, mix, cnts, f"loglik {loglik}")
    
            if useBic:
                loglik = common.bic(X,mix,loglik)
            if loglik > maxll:
                maxll = loglik
                maxseed = seed
                logliks[K] = loglik
                bestPlotArgs[K] = tuple([K,mix,cnts,loglik])
    
        print(f"K {K} seed {maxseed}, loglik={maxll}")
    
    print(f"max logliks {logliks}")
    for k, args in bestPlotArgs.items():
        common.plot(X, args[1], args[2], f"EM K={args[0]} : {args[3]}")

# emTest(X)
emTest(X, useBic=True)

def bicTest():
    gm, scnts = common.init(X,14,99)
    bic = common.bic(X,gm,-1000)
    print(f"bic {bic}")
    
#bicTest()

# TODO: Your code here
def kmeansTest(X):
    Ks = [1,2,3,4]
    seeds = [0,1,2,3,4]
    
    K = Ks[0]
    seed = seeds[0]
    lowest_costs = []
    bestPlotArgs = dict()
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
                bestPlotArgs[K] = tuple([K,ret_mixture,ret_post,cost])
            else:
                if cost < low_cost:
                    low_cost = cost
                    bestPlotArgs[K] = tuple([K,ret_mixture,ret_post,cost])
    
    
            #common.plot(X, ret_mixture, ret_post, f"cost {cost}")
        lowest_costs.append(low_cost)
        
    print(f"lowest costs {lowest_costs}")
    for k, args in bestPlotArgs.items():
        common.plot(X, args[1], args[2], f"KMeans K={args[0]} : {args[3]}")

#kmeansTest(X)


