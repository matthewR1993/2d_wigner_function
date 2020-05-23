import numpy as np
from math import factorial, exp, sqrt


def tmsv_state(l=10, z=0.3):
    s = np.zeros((l, l), dtype=np.complex128)
    for n in range(l):
        s[n, n] = z**n
    return np.sqrt(1 - np.abs(z)**2) * s


def coherent_state(l=10, alpha=1.0):
    s = np.zeros(l, dtype=np.complex128)
    for n in range(l):
        s[n] = alpha**n / sqrt(factorial(n))
    return exp(-abs(alpha)**2 / 2) * s


def prob_distr(rho):
    l = len(rho)
    p = np.zeros((l, l), dtype=float)
    for m in range(l):
        for n in range(l):
            p[m, n] = np.real(rho[m, n, m, n])
    return p
