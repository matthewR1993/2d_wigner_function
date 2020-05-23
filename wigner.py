import numpy as np
from scipy.special import genlaguerre
from math import factorial


def wigner(rho, grd=50, a=6):
    """
    Wigner function for the two-mode state.
    :param rho: A density matrix of two-mode state given as rho(p1, p2, p1_, p2_)
    with matrix elements |p1, p2><p1_, p2_|.
    :return:
    """
    xvec = np.linspace(-a, a, grd)

    X, Y = np.meshgrid(xvec, xvec)
    A = 0.5 * np.sqrt(2) * (X + 1.0j * Y)
    B = 4 * np.abs(A) ** 2
    W = np.zeros(np.shape(np.tensordot(A, A, axes=0)), dtype=complex)
    l = len(rho)

    for n1 in range(l):
        for n2 in range(l):
            for p1 in range(l):
                for p2 in range(l):
                    if np.abs(rho[n1, n2, p1, p2]) > 0.0:
                        if p1 < n1:
                            x1 = (-2 * np.conj(A)) ** (n1 - p1) * np.sqrt(factorial(p1) / factorial(n1)) * genlaguerre(
                                p1, n1 - p1)(B)
                        else:
                            x1 = (2 * A) ** (p1 - n1) * np.sqrt(factorial(n1) / factorial(p1)) * genlaguerre(
                                n1, p1 - n1)(B)
                        if p2 < n2:
                            x2 = (-2 * np.conj(A)) ** (n2 - p2) * np.sqrt(factorial(p2) / factorial(n2)) * genlaguerre(
                                p2, n2 - p2)(B)
                        else:
                            x2 = (2 * A) ** (p2 - n2) * np.sqrt(factorial(n2) / factorial(p2)) * genlaguerre(
                                n2, p2 - n2)(B)

                        W += 4 * rho[n1, n2, p1, p2] * (-1) ** (n1 + n2) * np.tensordot(x1, x2, axes=0)

    return np.real(W) * np.tensordot(np.exp(-B / 2), np.exp(-B / 2), axes=0) / np.pi
