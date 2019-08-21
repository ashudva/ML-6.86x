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
    n,d = X.shape
    mu, var, pi = mixture
    K = mu.shape[0]

    # Posterior probabilities & Normal matrix
    post = np.zeros((n, K), dtype=np.float64)

    # Compute normal dist. matrix: (N, K)
    pre_exp = (2*np.pi*var)**(d/2)

    # Using single loop to complete Normal matrix: faster than broadcasting in 3D
    for i in range(n):
        # Compute difference: will be (K,d) for each n
        dist = X[i, :] - mu
        # Norm: will be (K,) for each n
        norm = np.sum(dist**2, axis=1)
        # Exponent term of normal
        post[i, :] = np.exp(-norm/(2*var))

    # Normal matrix: (n, K)
    post = post/pre_exp

    numerator = post*pi
    # vector p(x;theta)
    denominator = np.sum(numerator, axis=1).reshape(-1, 1)

    # This is the matrix of posterior probs p(j|i)
    post = numerator/denominator

    log_lh = np.sum(np.log(denominator), axis=0).item()    # Log-likelihood

    return post, log_lh


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
    n,d = X.shape
    K = post.shape[1]

    nj = np.sum(post, axis=0)   # shape is (K, )

    pi = nj/n   # Cluster probs; shape is (K, )

    mu = (post.T @ X)/nj.reshape(-1,1)  # Revised means; shape is (K,d)

    norms = np.zeros((n, K), dtype=np.float64)   # Matrix to hold all the norms: (n,K)

    for i in range(n):
        dist = X[i,:] - mu
        norms[i,:] = np.sum(dist**2, axis=1)

    var = np.sum(post*norms, axis=0)/(nj*d)     # Revised variance; shape is (K, )

    return GaussianMixture(mu, var, pi)


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
    old_log_lh = None
    new_log_lh = None  # Keep track of log likelihood to check convergence

    # Start the main loop
    while old_log_lh is None or (new_log_lh - old_log_lh > 1e-6*np.abs(new_log_lh)):

        old_log_lh = new_log_lh

        # E-step
        post, new_log_lh = estep(X, mixture)

        # M-step
        mixture = mstep(X, post)

    return mixture, post, new_log_lh
