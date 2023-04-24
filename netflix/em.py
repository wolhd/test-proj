"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
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
    # ignore x[i] if 0
    def mv_pdf(x, mu, var):
        mpdf = 1
        for i in range(len(x)):
            if x[i] != 0:
                updf = uv_pdf(x[i], mu[i], var)
                mpdf = mpdf * updf
        return mpdf
    
    # xu (d,)
    def f_u_i(xu, mix, i):
        pi = mix.p[i]
        mpdf = mv_pdf(xu, mix.mu[i,:], mix.var[i])
        fui = np.log(pi) + np.log(mpdf)
        return fui
    
    K = len(mixture.p)
    U = X.shape[0]
    
    log_pjus = np.zeros((U,K))
    
    for u in range(U):
        for j in range(K):
            all_fui = np.zeros(K)
            for i in range(K):
                fui = f_u_i(X[u,:], mixture, i)
                all_fui[i] = fui
            sum_fuis = all_fui.sum()
            fuj = all_fui[j]
            log_pju = fuj - np.log(sum_fuis)
            log_pjus[u,j] = log_pju
            
            
            


def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    raise NotImplementedError


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


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    raise NotImplementedError
