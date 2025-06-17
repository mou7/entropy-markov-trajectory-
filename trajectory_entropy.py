"""
Author: Mohamed Kafsi

This code allows for computing the matrix of trajectory entropies associated with
an irreducible finite state Markov chain.

Definitions and computations can be found

  The entropy of Markov trajectories
  http://www-isl.stanford.edu/~cover/papers/paper101.pdf

The example of the MC presented in this script is from the paper

  The entropy of conditional Markov trajectories
  http://arxiv.org/abs/1212.2831

"""

import numpy as np
import scipy.linalg


def local_entropy(P):
    """Computes the local entropy at each state of the MC defined by the transition
    probabilities P"""
    if not isinstance(P, np.ndarray) or P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError("P must be a square numpy array")
    
    L = np.zeros_like(P)
    mask = P > 0
    L[mask] = np.log2(P[mask])
    K = np.dot(P, L.T)
    entropy_out = -1 * np.diagonal(K)
    return entropy_out.reshape((P.shape[0], 1))


def stationary_distribution(P):
    """Computes the stationary distribution mu associated with the MC whose
    transition probabilities are given by the numpy array P
    IMPORTANT: the MC must be irreducible and aperiodic to admit a stationary
    distribution
    """
    if not isinstance(P, np.ndarray) or P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError("P must be a square numpy array")
    
    # Check if rows sum to 1 (valid transition matrix)
    row_sums = np.sum(P, axis=1)
    if not np.allclose(row_sums, 1.0):
        raise ValueError("Transition matrix rows must sum to 1")
    
    # Find left eigenvector corresponding to eigenvalue 1
    eigenvalues, eigenvectors = scipy.linalg.eig(P.T, left=False, right=True)
    
    # Find the eigenvalue closest to 1
    idx = np.argmin(np.abs(eigenvalues - 1.0))
    
    if not np.isclose(eigenvalues[idx], 1.0, atol=1e-10):
        raise ValueError("No eigenvalue close to 1 found. Matrix may not be irreducible.")
    
    v = np.real(eigenvectors[:, idx])
    mu = np.abs(v)
    mu /= np.sum(mu)
    return mu


def trajectory_entropy(P):
    """Returns the matrix of trajectories entropy H associated to MC whose transition
    probabilities are given by the numpy array P.
    IMPORTANT: the MC is irreducible and aperiodic"""
    if not isinstance(P, np.ndarray) or P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError("P must be a square numpy array")
    
    n = P.shape[0]
    mu = stationary_distribution(P)
    A = np.tile(mu, (n, 1))
    # local entropies of the MC
    l_entropy = local_entropy(P)
    H_star = np.tile(l_entropy, (1, n))
    # entropy rate
    entropy_rate = np.dot(mu.T, l_entropy)
    H_delta = np.diagflat(entropy_rate / mu)
    
    # Use more stable matrix computation
    try:
        matrix_to_invert = np.identity(n) - P + A
        K = np.dot(np.linalg.inv(matrix_to_invert), H_star - H_delta)
    except np.linalg.LinAlgError:
        raise ValueError("Matrix inversion failed. The Markov chain may not be irreducible.")
    
    K_tilda = np.tile(np.diag(K).T, (n, 1))
    H = K - K_tilda + H_delta
    return H


if __name__ == "__main__":
    P = np.zeros((5, 5))
    P[0, 1], P[0, 2] = 0.25, 0.75
    P[1, 4] = 1
    P[2, 1], P[2, 3] = 0.5, 0.5
    P[3, 4] = 1
    P[4, 0], P[4, 3] = 0.5, 0.5
    
    np.set_printoptions(precision=3, suppress=True)
    print("Transition probability matrix \n {}".format(P))
    print("Row sums: {}".format(np.sum(P, axis=1)))  # Verify it's a valid transition matrix
    H = trajectory_entropy(P)
    print("Trajectory entropies matrix \n {}".format(H))
