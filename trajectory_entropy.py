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
    # TODO: Not optimal memory wise !!
    L = np.copy(P)
    L[P > 0] = np.log2(P[P > 0])
    K = np.dot(P, np.transpose(L))
    entropy_out = -1*np.diagonal(K)
    return entropy_out.reshape((P.shape[0], 1))


def stationary_distribution(P):
    """Computes the stationary distribution mu associated with the MC whose
    transition probabilities are given by the numpy array P
    IMPORTANT: the MC must be irreducile and aperiodic to admit a sttinary
    distribution
    """
    v = np.real(scipy.linalg.eig(P, left=True, right=False)[1][:, 0])
    mu = np.abs(v)/np.sum(np.abs(v))
    return mu


def trajectory_entropy(P):
    """Returns the matrix of trajectories entropy H associed to MC whose transition
    probabilities are given by the numpy array P.
    IMPORTANT: the MC is irreducile and aperiodic"""
    n = P.shape[0]
    mu = stationary_distribution(P)
    A = np.tile(mu, (n, 1))
    # local entropies of the MC
    l_entropy = local_entropy(P)
    H_star = np.tile(l_entropy, (1, n))
    # entropy rate
    entropy_rate = np.dot(mu.transpose(), l_entropy)
    H_delta = np.diagflat(entropy_rate/mu)
    K = np.dot(np.linalg.inv(np.identity(n) - P + A), H_star-H_delta)
    K_tilda = np.tile(np.diag(K).transpose(), (n, 1))
    H = K - K_tilda + H_delta
    return H


if __name__ == "__main__":
    P = np.zeros((5, 5))
    P[0, 1], P[0, 2] = 0.25, 0.75
    P[1, 4] = 1
    P[2, 1], P[2, 3] = 0.5, 0.5
    P[3, 4] = 1
    P[4, 0], P[4, 3] = 0.5, 0.5
    P[4, 0] = 0.5
    np.set_printoptions(precision=3, suppress=True)
    print("Transition probability matrix \n {}".format(P))
    H = trajectory_entropy(P)
    print("Trajectory entropies matrix \n {}".format(H))
