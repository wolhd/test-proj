"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture





def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    
    # univariate normal pdf
    def uv_pdf(x1, mu, var):
        pdf = 1/(var**0.5 * (2*np.pi)**0.5) * np.exp( -1/2 * (x1 - mu)**2/var)
        return pdf

    # multivariate normal pdf where covariance is diagonal of same variance   
    # size of x is (d,), mu is (d,), var is scalar
    def mv_pdf(x, mu, var):
        mpdf = 1
        for i in range(len(x)):
            updf = uv_pdf(x[i], mu[i], var)
            mpdf = mpdf * updf
        return mpdf
    
    
    # gauss params for K clusters
    mu_K = mixture.mu
    var_K = mixture.var
    p_K = mixture.p
    K = p_K.shape[0]
    N = X.shape[0]
    
    loglik = 0
    all_scount = np.zeros((N, K))
    
    for i in range(N):
        x = X[i,:]
        scount_ks = np.array([]) # soft count for all K
        for j in range(K):
            pji = p_K[j] * mv_pdf(x, mu_K[j], var_K[j])
            scount_ks = np.append(scount_ks, pji)
        pdf = scount_ks.sum()
        scount_ks = scount_ks / pdf
        all_scount[i] = scount_ks
        loglik = loglik + np.log(pdf)
    return all_scount, loglik



def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    # pji = post[i,j]
    #    p(j|i) prob i is from j
    # u_j = sum for all i (pji * x_i) / ( sum for all i pji ) 
    #    i.e., for a j,  weighted average of x's, weighted by its pji
    # p_j mixture weight is average of all pji across i's for this j
    #    p_j = 1/n * sum over all i's ( pji )
    # sigma_j = weighted average of mean-error-squared / d 
    #    weighted by pji, d is x dimension
    #    = sum over i's (pji * (x_i - u_j)^2) / (d * sum over i's (pji))
    
    # mixture sizes: mu (K,d), var (K,), p (K,)

    K = post.shape[1]
    d = X.shape[1]
    N = X.shape[0]
    mu = np.zeros((K,d))
    var = np.zeros((K,))
    p = np.zeros((K,))
    
    for j in range(K):
        pjis = post[:,[j]]  # (N,1)
        mu_j = pjis * X # (N,1) * (N,d) = (N,d)
        mu_j = mu_j.sum(axis=0) / pjis.sum() # (1,d) / (1) = (1,d)
        mu[j,:] = mu_j # (K,d) <- (1,d)
        
        p_j = pjis.sum() / N  # (1) / (1) = (1)
        p[j] = p_j  # (K,) <- (1)
        
        mean_err_2d = X - mu_j  # (N,d) - (1,d) = (N,d)
        mean_err = (mean_err_2d ** 2).sum(axis=1).reshape(-1,1) # (N,1)
        var_numer = pjis * mean_err # (N,1) * (N,1) = (N,1)
        var_numer = var_numer.sum() # (1)
        var_denom = d * pjis.sum() # (1) * (1)
        var_j = var_numer / var_denom
        var[j] = var_j # (K,) <- (1)
        
    gm = GaussianMixture(mu=mu, p=p, var=var)
    return gm    
        

def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    raise NotImplementedError
