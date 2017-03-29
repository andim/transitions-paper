import warnings
import numpy as np
import misc

## calculate long term growth rates
from cevolimmune import sumlog, zrecursionNstate 

def zstogrowthrate(zs, nburnin=1):
    nburnin = int(nburnin)
    data = zs[nburnin:]
    return sumlog(data)/data.shape[0]

def switching_matrix_3by3(x):
    """take 6 parameters in [0, 1] and return valid switching matrix (row stochastic)"""
    alpha, beta, gamma, delta, epsilon, zeta = x
    # define matrix in switching rates in a way that all parameters can be in [0, 1]
    pi = np.array([[alpha, (1.0-beta)*epsilon, (1.0-gamma)*zeta],
                    [(1.0-alpha)*delta, beta, (1.0-gamma)*(1.0-zeta)],
                    [(1.0-alpha)*(1.0-delta), (1.0-beta)*(1.0-epsilon), gamma]])
    return pi

def switching_matrix_2by2(x):
    """take 2 parameters in [0, 1] and return valid switching matrix (row stochastic)"""
    alpha, beta = x
    pi = np.array([[1.0-alpha, alpha],
                    [beta, 1.0-beta]])
    return pi

def steadystate3types(x):
    """steady state vector for three state markov chain as defined by switching_matrix_3by3"""
    alpha, beta, gamma, delta, epsilon, zeta = x
    vector = np.array([(-1 + beta) * (-1 + gamma) * (epsilon * (-1 + zeta) - zeta),
                         - (-1 + alpha) * (-1 + gamma) * (1 + (-1 + delta) * zeta),
                         (-1 + alpha) * (-1 + beta) * (-1 + delta * epsilon)])
    return vector / np.sum(vector)

def steadystate2types(x):
    return to_api(*x)[1]

def Lambda_f3(x, f1, f2, aenv, pienv, niter, nburnin, seed=None):
    """Lambda based on fitness vectors for three states,
    x: switching rates alpha, beta, gamma, delta, epsilon, zeta
    f1, f2: fitness of types in environment 1, 2
    """
    pi = switching_matrix_3by3(x)
    Ntypes = pi.shape[1]
    A = np.empty((2, Ntypes, Ntypes))
    A[0] = pi * f1
    A[1] = pi * f2
    return Lambda_matrix(A, aenv, pienv, niter, nburnin=nburnin, seed=seed)

def Lambda_f(pi, f1, f2, aenv, pienv, niter, nburnin, seed=None):
    """Lambda based on fitness vectors for two states,
    pi: switching matrix
    f1, f2: fitness of types in environment 1, 2
    """
    Ntypes = pi.shape[1]
    A = np.empty((2, Ntypes, Ntypes))
    A[0] = pi * f1
    A[1] = pi * f2
    return Lambda_matrix(A, aenv, pienv, niter, nburnin=nburnin, seed=seed)

def Lambda_matrix(A, aenv, pienv, niter=1000, seed=None, nburnin=1):
    alpha, beta = from_api(aenv, pienv)
    return zstogrowthrate(zrecursionNstate(A, alpha, beta, niter=niter, seed=seed), nburnin=nburnin)

## analytical optima

def pihat(pienv, fA, fB):
    "optimal mixture of individuals with fitness fs0, fs1"
    return (fB[1]*fB[0]-fB[1]*fA[0]*(1.0-pienv)-fB[0]*fA[1]*pienv)/((fB[0]-fA[0])*(fB[1]-fA[1]))

def plow(w1, w2, amin=0.0, amax=None):
    "lower bound on switching as a function of tau"
    if amax is None:
        amax = w2
    a = np.linspace(amin, amax)
    return (1.0-w1)*(1.0-a/w2)/((1.0/w2-w1)*(1.0-a)), to_tau(a) 

def phigh(w1, w2, amin=0.0, amax=None):
    "upper bound on switching as a function of tau"
    if amax is None:
        amax = w1
    a = np.linspace(amin, amax)
    return (1.0/w1-1)*(1.0-a*w2)/((1.0/w1-w2)*(1.0-a)), to_tau(a) 

## parameter conversions

def from_api(a, pi):
    "p, q of two state Markov chain from a, pi"
    return (1-a)*pi, (1-a)*(1-pi)

def to_api(p, q):
    "a, pi of two state Markov chain from p, q"
    # if q is not defined then p is 0 and pi 1 (and vice versa)
    return 1-p-q, np.where(p, np.where(q, p/(p+q), 1), 0)

def to_tau(a):
    "characteristic time tau from second eigenvalue a"
    a = np.asarray(a)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return 1.0 / np.log(1.0/a)

def from_tau(tau):
    "a from tau"
    tau = np.asarray(tau)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.exp(-1.0/tau)
